#! /usr/bin/env python


#
# Drone position degree of freedom in flat space. The dynamic equations for 
# position are linear in the flat space.
#
#
#

import os, signal


import torch
from torch.autograd import Function, Variable   
import torch.nn.functional as F 
from torch import nn
from torch.nn.parameter import Parameter
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.FloatTensor)

from mpc import util
#import control as ctl
import numpy as np


class DronePositionFlat(nn.Module):

    def __init__(self, f = 1., u_min = 0.0, u_max =1.0):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Settings for the MPC control problem
        #   u_min, u_max : lower and upper limits of the input
        #   n_state : number of states of the system
        #   n_ctrl : number of control inputs to the system
        #   goal_state : desired final output state for the system
        #   goal_weights : output performance weights
        #   ctrl_penalty : input effort weights
        #   mpc_eps :
        #   linesearch_decay : 
        #   max_linesearch_iter : 

        self.u_min = u_min
        self.u_max = u_max
        self.n_state = 4
        self.n_ctrl = 1
        self.goal_state = torch.tensor([[1.0],[0.0],[0.0],[0.0]]).to(self.device) 
        self.goal_weights = torch.tensor([100.0, 1.0, 1.0, 1.0]).to(self.device)
        self.ctrl_penalty = 1.0
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 10


        # continuous time dynamics, state space representation
        self.A = np.array([[0., 1., 0., 0.],     # system matrix
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 0., 0.]])

        self.B = np.array([[0.],                 # input matrix
                          [0.],
                          [0.],
                          [1.]])

        self.C = np.array([1.0, 1.0, 1.0, 1.0])      # output matrix
        self.D = np.array([0.0])                     # retransmission matrix

        #self.sys = ctl.StateSpace(A,B,C,D)

        # discrete time dynamics
        self.T = 1./f     
        #self.sysd = sys.sample(T)

        #self.Ad = np.array(sysd.A)
        #self.Bd = np.array(sysd.B)
        #self.Cd = np.array(sysd.C)
        #self.Dd = np.array(sysd.D)

        self.Ad = np.array([[1.0, 0.01, 0.00005, 0.000000166666667],
                            [0.0, 1.0, 0.01, 0.00005],
                            [0.0, 0.0, 1.0, 0.01],
                            [0.0, 0.0, 0.0, 1.0]])

        self.Bd = np.array([[0.000000000416666667],
                            [0.000000166666667],
                            [0.00005],
                            [0.01]])

        # convert matrices to torch.tensor and move to device
        self.A = torch.from_numpy(self.A).to(self.device)
        self.B = torch.from_numpy(self.B).to(self.device)
        self.C = torch.from_numpy(self.C).to(self.device)
        self.D = torch.from_numpy(self.D).to(self.device)

        self.Ad = torch.from_numpy(self.Ad).to(self.device)
        self.Bd = torch.from_numpy(self.Bd).to(self.device)
        self.Cd = self.C#torch.from_numpy(self.Cd).to(self.device)
        self.Dd = self.D#torch.from_numpy(self.Dd).to(self.device)


    def forward(self,x,u):
        """
        Forward pass implements the discrete dynamic equations of the system
        Args:
            x : state vector of 4 elements
            u : input vector of 1 element
        Returns:
            Updated state vector x after application of input u
        """

        # limit the control input to the lower and upper limits
        u = torch.clamp(u, self.u_min, self.u_max)
        x = torch.mm(self.Ad,x) + torch.mul(self.Bd,u)
        return x

    def cforward(self,x,u):
        """
        Forward pass implementing the continuous dynamic equations
        """
        u = torch.clamp(u, self.u_min, self.u_max)
        x = torch.mm(self.A,x) + torch.mul(self.B, u)
        return x



def main():
    import matplotlib.pyplot as plt

    sys = DronePositionFlat(f=100, u_min = -5, u_max= 5)
    
    x0 = torch.tensor([[1.],[0.],[0.],[0.]]).to(sys.device).type(torch.float64)
    u = torch.tensor([1.0]).to(sys.device).type(torch.float64)

    x_prev = x0
    # define simulation time
    t_max = 10
    dt = 1/100.
    t = np.arange(0.0,t_max,dt)

    states = []
    for i in t:
        x = sys(x_prev,u)
        x_prev = x

        states.append(x.cpu().numpy())

    fig = plt.figure(figsize=(20,10))
    fig.suptitle(" Discrete Position Dynamics, Drone")
    ax0 = fig.add_subplot(1,1,1)#, projection='3d')

    ax0.plot(t, [a[0] for a in states], label = 'position', color = 'r')
    ax0.plot(t, [a[1] for a in states], label = 'velocity', color = 'b')
    ax0.plot(t, [a[2] for a in states], label = 'acceleration', color = 'g')
    ax0.plot(t, [a[3] for a in states], label = 'jerk', color = 'c')

    ax0.set_title("Discrete system response", fontsize='small')
    ax0.legend(loc='upper right', shadow=True, fontsize='small')
    ax0.set_xlabel("time {s}")
    ax0.set_ylabel("{m},{m/s},{m/s2},{m/s3}")

    plt.show()    

if __name__ == '__main__':
    main()