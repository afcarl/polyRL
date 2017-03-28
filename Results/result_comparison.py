import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 5000
eps = range(eps)




def single_plot_episode_stats(stats, eps,  smoothing_window=200, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    ##Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Deep Q Learning on Cart Pole")


    plt.legend(handles=[cum_rwd,])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN on Cart Pole - Single Run - Larger Network (Layer 1, 512 Units, Layer 2, 256 Units)")
    plt.show()

    return fig



def comparison_plot(stats1, stats2, eps,  smoothing_window=100, noshow=False):

    ##Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="PolyRL DDPG")
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="DDPG")


    plt.legend(handles=[cum_rwd_1, cum_rwd_2])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Inverted Pendulum Environment (Continuous Action Space)")
    plt.show()

    return fig


ddpg = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/DDPG.npy')
polyrl_ddpg = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/PolyRL_DDPG.npy')

ddpg = ddpg[0:2000]
polyrl_ddpg = polyrl_ddpg[0:2000]

eps = 2000
eps = range(eps)


def main():
    # single_plot_episode_stats(ddpg, eps)

    comparison_plot(polyrl_ddpg, ddpg, eps)



if __name__ == '__main__':
    main()

