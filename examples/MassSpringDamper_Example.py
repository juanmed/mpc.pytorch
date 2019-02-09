#! /usr/bin/env python

###############################################################################
# PendulumExample.py
#
# Largely a script version of the Jupyter Notebook provided as an example in the source 
# repository.
#
# NOTE: Any plotting is set up for output, not viewing on screen.
#       So, it will likely be ugly on screen. The saved PDFs should look
#       better.
#
# Created: ~02/06/19
#   * Ben Armentor - bma8468@louisiana.edu
#   
# Modified:
#   * 02/09/19 - Joshua Vaughan - joshua.vaughan@louisiana.edu
#       - Added additional commenting
#       - Set up to plot state history, rather than save a video
# 
# TODO:
#   * 02/09/19 - JEV 
#       - Save state and control history for multi-batch solutions
#       - Set up plotting for mulit-batch solutions
#       - Improve commenting around the cost function setup
#       - Explore using the last solved control input for next solution
###############################################################################

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

# Note: These imports assume that the mpc.pytorch library is installed in your path or
# this file is either being run from the main mpc.pytorch folder 
# When in development, I recommend that latter. 
# You can do this can keep the file in the examples subfolder by using the command:
#  run /examples/MassSpringDamper_Example.py in IPython
#    or
#  python /examples/MassSpringDamper_Example.py from a normal terminal prompt
from mpc import util
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx.mass_spring_damper import MassSpringDamperDx

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


if __name__ == '__main__':
    m = 1.0                     # mass (kg)
    k = (2*np.pi)**2            # spring constant - (2*pi)^2 rad/s results in 1 Hz natural freq
    wn = np.sqrt(k / m)         # Define the natural frequency

    # Select damping ratio and use it to choose an appropriate c
    zeta = 0.00                 # damping ratio
    c = 2 * zeta * wn * m       # damping coeff.

    # Packing system parameters into a tensor
    params = torch.Tensor((m, c, k))
    
    # Create an instance of the MassSpringDamperDx object. 
    # See the file mass_spring_damper.py in env_dx for details on this class.
    dx = MassSpringDamperDx(params, simple=True)
    
    # You can adjust the upper and lower limits on torque below to see how they effect
    # the solution. For this example, you'll see that the solutions will sometimes not
    # converge to the optimal. The defaults for the environment are +/-20N. For the 
    # parameters chosen above, we'd need at least (k * goal_position) amount of force
    # to reach goal_position position of the mass.
    #
    # dx.lower = -5.0
    # dx.upper = 15.0

    # Parameters needed for the solution procedure
    n_batch = 1     # Number of batches to run
    T = 100         # Number of time steps to simulate/solve over
    mpc_T = 20      # Number of time steps in the MPC prediction horizon

    def uniform(shape, low, high):
        """ Defines a uniform distribution of random numbers 
        
        Arguments:
          shaped : shape of desired tensor
          low : minimum value to contain in the distribution
          high : maximum value to contain in the distribution
        """
        
        r = high-low
        
        return torch.rand(shape) * r + low


    torch.manual_seed(0) # Seed the random number generator for repeatable results
    
    # Define initial conditions for the simulation, using a uniform random distribution
    x = uniform(n_batch, -1.0, 1.0)
    x_dot = uniform(n_batch, -1.0, 1.0)
    
    # Stack the state initial conditions into a PyTorch tensor
    x_init = torch.stack((x, x_dot), dim=1)

    x = x_init
    u_init = None

    # The displacement term is weight 10x that of the velocity term. We care more about
    # reaching the desired position than velocity.
    #
    # These would be values to "play" with to develop some intuition about how 
    # to design cost functions, etc.
    goal_weights = torch.Tensor((1.0, 0.1))

    # The desired state is 0.5m displacement of the mass.
    # Goal is:
    #    x = 0.5
    #    x_dot = 0
    goal_position = 0.5     # m
    goal_velocity = 0.0     # m/s
    goal_state = torch.Tensor((goal_position, goal_velocity))

    # The penalty on control is 1/1000000 of that on the displacement terms of the state
    # vector. This would be another value to "play" with to understand the 
    # relationship between the various elements in the cost function
    #
    # Here, it's so low that we basically aren't penalizing control effort. This is done
    # because we need a steady-state force to reach our goal position. If we overly 
    # penalize force in the cost function, then that steady-state force will "cost" too
    # much to allow us to get near our goal position.
    ctrl_penalty = 1e-6
    
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(dx.n_ctrl)
    ))
    
    # TODO: 02/09/19 - JEV - clarify my understanding of these operations and comment
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl)))
    
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1
    )
    
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)


    # Set up NumPy arrays to allow us to plot the state and control history
    # We'll only grab the first of the batch for now. 
    # TODO: 02/09/19 - JEV - Include others from the multi-batch solutions
    response = np.zeros((T, dx.n_state))
    control_inputs = np.zeros((T, dx.n_ctrl))
    time = np.arange(0, T * dx.dt, dx.dt)
    

    # Now, we actually solve over T number of timesteps
    # tqdm just gives us a nice progress meter of the solution loop
    for timestep in tqdm(range(T)):
        # Each call to the MPC solver returns the state history, the control history (actions),
        # and value of the objective function for the solution over the mpc_T number of 
        # timesteps.
        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
            dx.n_state,             # Number of states
            dx.n_ctrl,              # Number of control inputs
            mpc_T,                  # MPC prediction horizon in number of timesteps
            u_init=u_init,          # Initial guess for inputs
            u_lower=dx.lower,       # Lower limit on inputs
            u_upper=dx.upper,       # Upper limit on inputs
            lqr_iter=50,            # Number of iterations per LQR solution step
            verbose=0,              # Verbosity, 0 is just warnings. 1 will give more info
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=dx.linesearch_decay,
            max_linesearch_iter=dx.max_linesearch_iter,
            grad_method=GradMethods.AUTO_DIFF,
            eps=1e-2,
        )(x, QuadCost(Q, p), dx)
        
        # Save the first of the nominal actions determined by the MPC solution to use as 
        # the real next control input
        next_action = nominal_actions[0]
        
        # Update the initial control input to include the current input as the first in 
        # the sequence, then zero the rest
        # TODO: 02/09/19 - JEV - Would be better to use the previous solution as the 
        # initial guess here? The mpc.MPC function also has a prev_ctrl argument to explore.
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
        u_init[-2] = u_init[-3]
        
        # Calling the module causes the function forward to be called, while
        # taking care of running registered hooks. See:
        #   https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward
        # For the system here, the equations of motion are run in the forward
        # pass, so this basically just updates the states for the next timestep.
        x = dx(x, next_action)
        
        # TODO: 02/09/19 -JEV - update to get more than just the first batch's solution
        response[timestep,:] = x[0].detach().numpy()
        control_inputs[timestep,:] = next_action[0].detach().numpy()


    # Now, plot the response and control input
    
    # Response 
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

    # Change the axis units font
    plt.setp(ax.get_ymajorticklabels(),fontsize=18)
    plt.setp(ax.get_xmajorticklabels(),fontsize=18)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Turn on the plot grid and set appropriate linestyle and color
    ax.grid(True,linestyle=':', color='0.75')
    ax.set_axisbelow(True)

    # Define the X and Y axis labels
    plt.xlabel('Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel('Position (m)', fontsize=22, weight='bold', labelpad=10)
 
    plt.plot(time, response[:,0], linewidth=2, linestyle='-', label=r'Angle')

    # uncomment below and set limits if needed
    # plt.xlim(0,5)
    # plt.ylim(0,10)

    # Create the legend, then fix the fontsize
    # leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    # ltext  = leg.get_texts()
    # plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    # save the figure as a high-res pdf in the current folder
    plt.savefig('MassSpringDamper_MPC_PositionResponse.pdf')

    # show the figure
    # plt.show()
    
    # Control Input
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

    # Change the axis units font
    plt.setp(ax.get_ymajorticklabels(),fontsize=18)
    plt.setp(ax.get_xmajorticklabels(),fontsize=18)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Turn on the plot grid and set appropriate linestyle and color
    ax.grid(True,linestyle=':', color='0.75')
    ax.set_axisbelow(True)

    # Define the X and Y axis labels
    plt.xlabel('Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel('Force (N)', fontsize=22, weight='bold', labelpad=10)
 
    plt.plot(time, control_inputs, linewidth=2, linestyle='-', label=r'Force')

    # uncomment below and set limits if needed
    # plt.xlim(0,5)
    # plt.ylim(0,10)

    # Create the legend, then fix the fontsize
    # leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    # ltext  = leg.get_texts()
    # plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    # save the figure as a high-res pdf in the current folder
    plt.savefig('MassSpringDamper_MPC_ControlInput.pdf')

    # show the figure
    # plt.show()
    
    plt.close('all')
