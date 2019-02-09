#! /usr/bin/env python

###############################################################################
# PendulumExample.py
#
# Largely a script version of the Jupyter Notebook provided as an example in the source repository.
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
#  run /examples/PendulumExample.py in IPython
#    or
#  python /examples/PendulumExample.py from a normal terminal prompt
from mpc import util
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx.pendulum import PendulumDx

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


if __name__ == '__main__':
    g = 9.81        # gravity (m/s)
    l = 1.0         # length of the pendulum (m)
    m = 1.0         # mass of the pendulum (kg)
    
    # Pack the pendulum parameters into a PyTorch Tensor
    params = torch.Tensor((g, m, l))
    
    # Create an instance of the PendulumDx object. See the file pendulum.py in env_dx for
    # details on this class.
    dx = PendulumDx(params, simple=True)
    
    # You can adjust the upper and lower limits on torque below to see how they effect
    # the solution. For this example, you'll see that the solutions will sometimes not
    # converge to the optimal. The defaults for the pendulum swingup environment are +/-2
    # dx.lower = -3.0
    # dx.upper = 0.0

    # Parameters needed for the solution procedure
    n_batch = 1     # Number of batches to run
    T = 100         # Number of time steps to simulate/solve over
    mpc_T = 50      # Number of time steps in the MPC prediction horizon

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
    th = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)
    thdot = uniform(n_batch, -1.0, 1.0)
    
    # Stack the state initial conditions into a PyTorch tensor
    # TODO: 02/09/19 - JEV - Why use cosine and sine here? Is it just a convenient way to 
    #                        normalize the states to the range +/-1?
    xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)

    x = xinit
    u_init = None

    # The cost terms for the swingup task can be alternatively obtained
    # for this pendulum environment with:
    # q, p = dx.get_true_obj()

    # We can choose between swinging up the pendulum to vertical or causing it to spin
    mode = 'swingup'
    # mode = 'spin'

    if mode == 'swingup':
        # In swing up mode, the displacement terms are weighted equally at 10x the angular
        # velocity.
        # These would be values to "play" with to develop some intuition about how 
        # to design cost functions, etc.
        goal_weights = torch.Tensor((1.0, 1.0, 0.1))

        # The desired state is vertical. As the angles and states are defined here, that
        # means the first state, cos(theta)=-1 and all others are zero
        # Goal is:
        #    cos(theta) = -1, theta = pi
        #    sin(theta) = 0, theta = pi
        #    theta_dot = 0
        goal_state = torch.Tensor((-1.0, 0.0, 0.0))

        # The penalty on control is 1/1000 of that on the displacement terms of the state
        # vector. This would be another value to "play" with to understand the 
        # relationship between the various elements in the cost function
        ctrl_penalty = 0.001
        
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

    elif mode == 'spin':
        Q = 0.001*torch.eye(dx.n_state+dx.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(
            mpc_T, n_batch, 1, 1
        )
        p = torch.tensor((0.0, 0.0, -1.0, 0.0))
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

    # Parse the cosine state to get the angle in degrees
    theta = np.rad2deg(np.arccos(response[:,0]))    # Units will be degrees
    theta_dot = response[:,2]                       # Units will be rad/s


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
    plt.ylabel('Angle (deg)', fontsize=22, weight='bold', labelpad=10)
 
    plt.plot(time, theta, linewidth=2, linestyle='-', label=r'Angle')

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
    plt.savefig('PendlumSwingup_MPC_AngleResponse.pdf')

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
    plt.ylabel('Torque (Nm)', fontsize=22, weight='bold', labelpad=10)
 
    plt.plot(time, control_inputs, linewidth=2, linestyle='-', label=r'Torque')

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
    plt.savefig('PendlumSwingup_MPC_ControlInput.pdf')

    # show the figure
    # plt.show()
    
    plt.close('all')
