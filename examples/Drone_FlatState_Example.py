import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from mpc import util
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

from drone_flat_state import DronePositionFlat as DroneFlat

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def graphs(time, response, control_inputs):
    # Response 
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # Change the axis units font
    #plt.setp(ax.get_ymajorticklabels(),fontsize=18)
    #plt.setp(ax.get_xmajorticklabels(),fontsize=18)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Turn on the plot grid and set appropriate linestyle and color
    ax.grid(True,linestyle=':', color='0.75')
    ax.set_axisbelow(True)

    # Define the X and Y axis labels
    ax.set_xlabel('Time (s)', fontsize=22, weight='bold', labelpad=5)
    ax.set_ylabel('Position {m}, Input {m/s4}', fontsize=22, weight='bold', labelpad=10)
 
    ax.plot(time, control_inputs, linewidth=2, linestyle='-', label='Snap', color = 'r')
    ax.plot(time, [a[0] for a in response], linewidth=2, linestyle='-', label='Position', color = 'b')

    # uncomment below and set limits if needed
    # plt.xlim(0,5)
    # plt.ylim(0,10)

    # Create the legend, then fix the fontsize
    ax.legend(loc='upper right', ncol = 1, fancybox=True)
    # ltext  = leg.get_texts()
    # plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    #plt.tight_layout(pad=0.5)

    # save the figure as a high-res pdf in the current folder
    plt.show()
    #plt.close('all')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    freq = 100. # hertz
    sys = DroneFlat(f = freq, u_min =-5., u_max = 10.)   # min, max must be double!

    # MPC problem parameters
    n_batch = 1         # number of batches...? not sure what is this
    t_max = 1.         # simulation time (seconds)
    dt = 1./freq        # time step size
    time = np.arange(0.0,t_max,dt) 
    mpc_T = 10          # MPC Prediction horizon

    
    x = torch.tensor([[0.,0.,0.,0.]]).to(sys.device)                        # initial state
    u_init = None                                                            # initial input

    goal_state = torch.tensor([1.0,0.0,0.0,0.0]).to(sys.device)                 # desired final state
    goal_weights = torch.tensor([1.0, 1.0e-3, 1.0e-3, 1.0e-3]).to(sys.device)                  # output performance weights
    ctrl_penalty = 1.0e-9                                                           # control effort penalty
    mpc_eps = 1e-3  
    linesearch_decay = 0.2
    max_linesearch_iter = 10

    # create state and input weigth matrices
    q = torch.cat((goal_weights, torch.tensor([ctrl_penalty]).to(device)))
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(sys.n_ctrl).to(device)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)
    
    response = []
    control_inputs = []

    for t in tqdm(time):

        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
            sys.n_state,             # Number of states
            sys.n_ctrl,              # Number of control inputs
            mpc_T,                  # MPC prediction horizon in number of timesteps
            u_init=u_init,          # Initial guess for inputs
            u_lower=sys.u_min,       # Lower limit on inputs
            u_upper=sys.u_max,       # Upper limit on inputs
            lqr_iter=100,            # Number of iterations per LQR solution step
            verbose=0,              # Verbosity, 0 is just warnings. 1 will give more info
            exit_unconverged=False,
            detach_unconverged=False,
            backprop=True,
            linesearch_decay=linesearch_decay,
            max_linesearch_iter=max_linesearch_iter,
            grad_method=GradMethods.ANALYTIC, # FINITE_DIFF,
            eps=mpc_eps,
        )(x, QuadCost(Q, p), sys)

        next_action = nominal_actions[0]
        #print("Actions: ",nominal_actions,nominal_actions.shape)

        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, sys.n_ctrl).to(device)), dim=0)
        u_init[-2] = u_init[-3]

        x = sys(x, next_action)
        #print("state",x,x.shape,x.ndimension())

        # TODO: 02/09/19 -JEV - update to get more than just the first batch's solution
        response.append(x[0].detach().cpu().numpy())
        control_inputs.append(next_action[0].detach().cpu().numpy())


    # Now, plot the response and control input
    
    graphs(time, response, control_inputs)

if __name__ == '__main__':
    main()