#! /usr/bin/env python

###############################################################################
# mass_spring_damper.py
#
# Simple mass-spring-damper example environment for use with mpc.pytorch
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
#
# TODO:
#   * 
###############################################################################

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MassSpringDamperDx(nn.Module):
    def __init__(self, params=None, simple=True):
        super().__init__()
        self.simple = simple
        
        # Define the timestep size for the simulation (s)
        self.dt = 0.05
        
        # Define the number of states. Here they are x and x_dot
        self.n_state = 2
        
        # Define the number of control inputs. Here, there is only the force on the mass
        self.n_ctrl = 1
        
        # Set the maximum force for the control input (N)
        # TODO: 02/09/19 - We should probably include this as a parameter
        self.max_force = 20.0

        if params is None:
            if simple:
                # m (kg), c (Ns/m), k (N/m)
                self.params = Variable(torch.Tensor((1.0, 0, (2*np.pi)**2)))
            else:
                # m (kg), c (Ns/m), k (N/m), zeta, wn (rad/s)
                self.params = Variable(torch.Tensor((1.0, 0, (2*np.pi)**2, 0, np.sqrt((2*np.pi)**2/1.0))))
        else:
            self.params = params

        # Check that we have the correct number of parameters
        assert len(self.params) == 3 if simple else 5

        # The goal state is to displace the mass by 1m and stop there (0 velocity)
        self.goal_state = torch.Tensor([1.0, 0.])
        
        # The velocity and displacement are penalized equally
        self.goal_weights = torch.Tensor([1.0, 1.0])
        
        # And control is penalized at 1/1000 of that value
        self.ctrl_penalty = 0.001
        
        # Set the lower and upper limits for the force input based on the max_force
        self.lower, self.upper = -self.max_force, self.max_force

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5


    def forward(self, x, u):
        """ 
        This forward pass seems to just mostly implement the equations of motion for the
        system.
        
        Arguments:
          x : State vector as pytorch tensor
          u : input vector
          
        Returns:
          Updated state tensor
        """
        
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        # Check the all the dimensions are correct, raise an AssertionError if not
        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.ndimension() == 2

        # Check if running on GPU with CUDA
        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        if not hasattr(self, 'simple') or self.simple:
            m, c, k = torch.unbind(self.params)
        else:
            m, c, k, zeta, wn = torch.unbind(self.params)

        # limit the control input to the lower and upper limits
        u = torch.clamp(u, self.lower, self.upper)[:,0]

        # Get the current states from the tensor
        x, x_dot = torch.unbind(x, dim=1)

        # Update the velocity
        if not hasattr(self, 'simple') or self.simple:
            # (Force - Damping - Spring)/Mass = x_Double_Dot
            x_dot = x_dot + self.dt * ((u - c * x_dot - k * x)/m)

        # Then, update the position
        x = x + self.dt * x_dot
        
        # And, stack them back into the state tensor for return
        state = torch.stack((x, x_dot), dim=1)

        if squeeze:
            state = state.squeeze(0)
            
        return state


    def get_frame(self, state):
        """ Get a frame to use in generating a video of the results """
        state = util.get_data_maybe(state.view(-1))
        
        # Check that we got the right number of states
        assert len(state) == 2
        
        # Parse the current states from the state tensor
        x, x_dot = torch.unbind(state)
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        plt.axis('equal') # This will make the distances represented on the axes equal
        
        # Add a rectangle representing the mass
        cart = mpatches.Rectangle((x - 0.2, 0 - 0.1), 0.4, 0.2, zorder=3)
        ax.add_patch(cart)
        
        ax.set_xlim((-3., 3.))
        ax.set_ylim((-2., 2.))
        
        return fig, ax


if __name__ == '__main__':

    # Define the parameters for simluation
    m = 1.0                      # mass (kg)
    k = (2*2*np.pi)**2           # spring constant (N/m)

    wn = np.sqrt(k/m)            # natural frequency (rad/s)

    # Select damping ratio and use it to choose an appropriate c
    zeta = 0.05                   # damping ratio
    c = 2*zeta*wn*m               # damping coeff.

    # Packing system parameters into a tensor
    params = torch.Tensor((m, c, k))
    dx = MassSpringDamperDx(params, simple=True)

    n_batch, T = 1, 100

    # Set up NumPy arrays to allow us to plot the state and control history
    # We'll only grab the first of the batch for now. 
    # TODO: 02/09/19 - JEV - Include others from the multi-batch solutions
    response = np.zeros((T, dx.n_state))
    control_inputs = np.zeros((T, dx.n_ctrl))
    time = np.arange(0, T * dx.dt, dx.dt)
    
    u = torch.zeros(T, n_batch, dx.n_ctrl)      # The control inputs are all zero
    
    xinit = torch.zeros(n_batch, dx.n_state)    # Initial conditions are zero, except
    
    # The displacement 
    xinit[:,0] = 0.75
    
    x = xinit
    
    # Just loop over T number of timesteps given the initial condition response
    for t in range(T):
        x = dx(x, u[t])
        
        # Save a png of the current timestep
        fig, ax = dx.get_frame(x[0])
        fig.savefig('massSpringDamper{:03d}.png'.format(t), dpi=300)
        plt.close(fig)

    # Now use the ffmpeg to create a video from those pngs
    vid_file = 'mass_spring_damper_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('{} -loglevel quiet '
            '-r 20 -f image2 -i massSpringDamper%03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}').format(
        FFMPEG_BIN,
        vid_file
    )
    os.system(cmd)
    
    # Then, delete the pngs
    for t in range(T):
        os.remove('massSpringDamper{:03d}.png'.format(t))
