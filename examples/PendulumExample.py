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
#   * 02/08/19 - Joshua Vaughan - joshua.vaughan@louisiana.edu
#       - Added additional commenting
#       - Set up to plot state history, rather than save a video
#
#
# TODO:
#   * 
###############################################################################

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from mpc import util
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

from tqdm import tqdm


if __name__ == '__main__':
    params = torch.Tensor((9.81, 1., 1.))
    dx = PendulumDx(params, simple=True)

    n_batch, T, mpc_T = 1, 100, 20

    def uniform(shape, low, high):
        r = high-low
        return torch.rand(shape)*r+low

    torch.manual_seed(0)
    th = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)
    thdot = uniform(n_batch, -1., 1.)
    xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)

    x = xinit
    u_init = None

    # The cost terms for the swingup task can be alternatively obtained
    # for this pendulum environment with:
    # q, p = dx.get_true_obj()

    mode = 'swingup'
    # mode = 'spin'

    if mode == 'swingup':
        goal_weights = torch.Tensor((1., 1., 0.1))
        goal_state = torch.Tensor((1., 0. ,0.))
        ctrl_penalty = 0.001
        q = torch.cat((
            goal_weights,
            ctrl_penalty*torch.ones(dx.n_ctrl)
        ))
        px = -torch.sqrt(goal_weights)*goal_state
        p = torch.cat((px, torch.zeros(dx.n_ctrl)))
        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
            mpc_T, n_batch, 1, 1
        )
        p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)
    elif mode == 'spin':
        Q = 0.001*torch.eye(dx.n_state+dx.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(
            mpc_T, n_batch, 1, 1
        )
        p = torch.tensor((0., 0., -1., 0.))
        p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

    for t in tqdm(range(T)):
        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
            dx.n_state, dx.n_ctrl, mpc_T,
            u_init=u_init,
            u_lower=dx.lower, u_upper=dx.upper,
            lqr_iter=50,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=dx.linesearch_decay,
            max_linesearch_iter=dx.max_linesearch_iter,
            grad_method=GradMethods.AUTO_DIFF,
            eps=1e-2,
        )(x, QuadCost(Q, p), dx)
        
        next_action = nominal_actions[0]
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
        u_init[-2] = u_init[-3]
        
        # Calling the module causes the function forward to be called, while
        # taking care of running registered hooks. See:
        #   https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward
        # For the system here, the equations of motion are run in the forward
        # pass, so this basically just updates the states for the next timestep.
        x = dx(x, next_action)


