#! /usr/bin/env python

###############################################################################
# cartpole.py
#
# Edits to cartpole environment script. Commenting to add clarity (mostly for myself)
# 
# Original repository links:
#  * https://locuslab.github.io/mpc.pytorch/
#  * https://github.com/locuslab/mpc.pytorch
#  * https://arxiv.org/abs/1703.00443
#  * https://arxiv.org/abs/1810.13400
#
# NOTE: Any plotting is set up for output, not viewing on screen.
#       So, it will likely be ugly on screen. The saved PDFs should look
#       better.
#
# Created: 02/09/19
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - http://www.ucs.louisiana.edu/~jev9637
#
# Modified:
#   * 
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
# plt.style.use('bmh')

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class CartpoleDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        # Define the number of states. They are, in order:
        # x         - The horizontal position of the cart 
        # dx        - The cart velocity
        # cos(th)   - Cosine of the pole angle, theta=0 is vertical up (noon on a clock)
        # sin(th)   - Sine of the pole angle, theta=0 is vertical up (noon on a clock)
        # dth       - Angular velocity of the pole
        # TODO: 02/09/19 - JEV - Why use cosine and sine rather than the angle directly. 
        #                        Is it to normalize the states to +/-1 bounds?
        self.n_state = 5

        
        # Define the number of control inputs - It is a force on the cart
        self.n_ctrl = 1

        # Set up the model parameters
        # gravity, masscart, masspole, length
        if params is None:
            # If none are passed used the defaults
            self.params = Variable(torch.Tensor((9.81, 1.0, 0.1, 0.5)))
        else:
            self.params = params
        
        # Check the there are the correct number of parameters
        assert len(self.params) == 4
        
        
        # Set up threholds on state values
        self.theta_threshold_radians = np.pi #12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.max_velocity = 10

        # The size of the timestep in simulation (s)
        self.dt = 0.05

        # Define the upper and lower limits for the control inputs
        self.force_mag = 100.0
        self.lower = -self.force_mag
        self.upper = self.force_mag
        
        # The goal state is for the pole to be vertical - theta = 0, so cos(theta)=1
        self.goal_state = torch.Tensor([ 0.0,  0.0, 1.0, 0.0, 0.0])
        
        # The angular displacement states are weighted 10x more than others
        self.goal_weights = torch.Tensor([0.1, 0.1, 1.0, 1.0, 0.1])
        
        # Control effort is weighted 1/1000 that of the angular displacement states
        self.ctrl_penalty = 0.001

        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2


    def forward(self, state, u):
        """ 
        This forward pass seems to just mostly implements the equations of motion for the
        system.
        
        Arguments:
          state : State vector as pytorch tensor
          u : input vector
          
        Returns:
          Updated state vector as pytorch tensor
        """
        
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        # Check if running on GPU with CUDA
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        
        # Parse out the parameters of the model from the params tensor
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length

        # force the control input to within the acceptable bounds
        u = torch.clamp(u[:,0], -self.force_mag, self.force_mag)

        # Parse out the current states of the model from the state tensor
        x, dx, cos_th, sin_th, dth = torch.unbind(state, dim=1)
        
        th = torch.atan2(sin_th, cos_th)

        cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
        
        # Calculate the angular acceleration and cart acceleration using the equations
        # of motion for the system
        th_acc = (gravity * sin_th - cos_th * cart_in) / \
                 (length * (4./3. - masspole * cos_th**2 /
                                     total_mass))
        xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

        # Now, update the states
        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc

        # And stack them back into a tensor to return
        state = torch.stack((
            x, dx, torch.cos(th), torch.sin(th), dth
        ), 1)

        return state

    def get_frame(self, state):
        """ Get a frame to use in generating a video of the results """
        state = util.get_data_maybe(state.view(-1))
        assert len(state) == 5
        
        # Parse the current states from the state tensor
        x, dx, cos_th, sin_th, dth = torch.unbind(state)
        gravity, masscart, masspole, length = torch.unbind(self.params)
        
        th = np.arctan2(sin_th, cos_th)
        th_x = sin_th * length*2
        th_y = cos_th * length*2
        
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        plt.axis('equal') # This will make the distances represented on the axes equal
        
        # Add a rectangle representing the cart
        cart = mpatches.Rectangle((x-0.2, 0-0.1), 0.4, 0.2, zorder=3)
        ax.add_patch(cart)
        
        # Then, plot the pole
        ax.plot((x, x+th_x), (0, th_y), color='k', zorder=10)
        
        ax.set_xlim((-3., 3.))
        ax.set_ylim((-2., 2.))
        
        return fig, ax

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


if __name__ == '__main__':
    dx = CartpoleDx()
    n_batch, T = 1, 100
    
    u = torch.zeros(T, n_batch, dx.n_ctrl)      # The control inputs are all zero
    
    xinit = torch.zeros(n_batch, dx.n_state)    # Initial conditions are zero, except
    
    # The states representing the pole angle 
    th = np.deg2rad(1) # not quite vertical
    xinit[:,2] = np.cos(th)
    xinit[:,3] = np.sin(th)
    
    x = xinit
    
    # Just loop over T number of timesteps given the initial condition response
    for t in range(T):
        x = dx(x, u[t])
        
        # Save a png of the current timestep
        fig, ax = dx.get_frame(x[0])
        fig.savefig('cartpole{:03d}.png'.format(t), dpi=300)
        plt.close(fig)

    # Now use the ffmpeg to create a video from those pngs
    vid_file = 'cartpole_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('{} -loglevel quiet '
            '-r 20 -f image2 -i cartpole%03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}').format(
        FFMPEG_BIN,
        vid_file
    )
    os.system(cmd)
    
    # Then, delete the pngs
    for t in range(T):
        os.remove('cartpole{:03d}.png'.format(t))
