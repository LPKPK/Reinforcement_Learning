import matplotlib.pyplot as plt
from env import BanditEnv
from tqdm import trange
from agent import EpsilonGreedy, UCB
import numpy as np


def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    r_total = []  # list of step rewards

    for i in range(k):  # bandits
        r_step = []

        for j in range(num_samples):
            r = env.step(i)  # reward on each step
            r_step.append(r)

        r_total.append(r_step)

    plt.xlabel('Reward Distribution')
    plt.ylabel('Action')

    plt.violinplot(r_total, showmedians=1)
    plt.show()


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # initialize env and agents here
    env = BanditEnv(k)

    e0 = EpsilonGreedy(k, 0, 0) # args: (number of arms, initial Q-values, epsilon)
    e1 = EpsilonGreedy(k, 0, 0.1)
    e2 = EpsilonGreedy(k, 0, 0.01)


    agents = [e0, e1, e2]

    rw = []     # rewards
    oa = []    # optimal rewards
    upper_bound = []
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        best_action = np.argmax(env.means)
        upper_bound.append(np.max(env.means))

        rw_trail = []
        oa_trail = []

        for agent in agents:
            agent.reset()
            rw_agent = []
            oa_agent = []

        # For each trial, perform specified number of steps for each type of agent
            for step in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
            
                if action == best_action:    # optimal action recorded
                    oa_agent.append(1)
                else:
                    oa_agent.append(0)

                rw_agent.append(reward)   # reward recorded

            rw_trail.append(rw_agent)         # agents
            oa_trail.append(oa_agent)

        rw.append(rw_trail)                   # trails
        oa.append(oa_trail)

    rw_avr = np.average(rw, 0)
    oa_avr = np.average(oa, 0)
    up_avr = np.mean(upper_bound)

    y_rw_err = []
    rw_std = np.std(rw, 0)
    up_err_std = np.std(upper_bound, 0)
    up_err = 1.96 * (up_err_std/np.sqrt(trials))

    for p in range(len(agents)):
        err = 1.96 * (rw_std[p]/np.sqrt(trials))
        y_rw_err.append(err)

    # Upper Bound Plot
    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

    x = np.arange(steps)
    
    plt.axhline(y = up_avr, linestyle='--', label='upper bound')
    plt.fill_between(x, up_avr + up_err, up_avr - up_err, alpha = 0.2)

    # Average Reward Plots
    # List of labels and colors
    labels = ['ε = 0', 'ε = 0.1', 'ε = 0.01']
    colors = ['b', 'g', 'r']

    for p in range(len(agents)):
        plt.plot(rw_avr[p], label=labels[p], color = colors[p])

    for p in range(len(agents)):
        plt.fill_between(x, rw_avr[p] + y_rw_err[p], rw_avr[p] - y_rw_err[p], alpha = 0.3 , color = colors[p])

    plt.legend()

    # Optimal Action Plots
    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action')

    for p in range(len(agents)):
        plt.plot(oa_avr[p], label=labels[p], color = colors[p])

    plt.legend()
    plt.show()




def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k)

    e0 = EpsilonGreedy(k, 0, 0, 0.1) # args: (number of arms, initial Q-values, epsilon)
    e1 = EpsilonGreedy(k, 5, 0, 0.1)
    e2 = EpsilonGreedy(k, 0, 0.1, 0.1)
    e3 = EpsilonGreedy(k, 5, 0.1, 0.1)
    u1 = UCB(k, 0, 2, 0.1)


    agents = [e0, e1, e2, e3, u1]

    rw = []     # rewards
    oa = []    # optimal rewards
    upper_bound = []
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        best_action = np.argmax(env.means)
        upper_bound.append(np.max(env.means))

        rw_trail = []
        oa_trail = []

        for agent in agents:
            agent.reset()
            rw_agent = []
            oa_agent = []

        # For each trial, perform specified number of steps for each type of agent
            for step in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
            
                if action == best_action:    # optimal action recorded
                    oa_agent.append(1)
                else:
                    oa_agent.append(0)

                rw_agent.append(reward)   # reward recorded

            rw_trail.append(rw_agent)         # agents
            oa_trail.append(oa_agent)

        rw.append(rw_trail)                   # trails
        oa.append(oa_trail)

    rw_avr = np.average(rw, 0)
    oa_avr = np.average(oa, 0)
    up_avr = np.mean(upper_bound)

    y_rw_err = []
    rw_std = np.std(rw, 0)
    up_err_std = np.std(upper_bound, 0)
    up_err = 1.96 * (up_err_std/np.sqrt(trials))

    for p in range(len(agents)):
        err = 1.96 * (rw_std[p]/np.sqrt(trials))
        y_rw_err.append(err)

    # Upper Bound Plot
    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

    x = np.arange(steps)
    
    plt.axhline(y = up_avr, linestyle='--', label='upper bound')
    plt.fill_between(x, up_avr + up_err, up_avr - up_err, alpha = 0.2)

    # Average Reward Plots
    # List of labels and colors
    labels = ['Q1 = 0, ε = 0', 'Q1 = 5, ε = 0', 'Q1 = 0, ε = 0.1', 'Q1 = 5, ε = 0.1', 'UCB c=2']
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']

    for p in range(len(agents)):
        plt.plot(rw_avr[p], label=labels[p], color = colors[p])

    for p in range(len(agents)):
        plt.fill_between(x, rw_avr[p] + y_rw_err[p], rw_avr[p] - y_rw_err[p], alpha = 0.3 , color = colors[p])

    plt.legend()

    # Optimal Action Plots
    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action')

    for p in range(len(agents)):
        plt.plot(oa_avr[p], label=labels[p], color = colors[p])

    plt.legend()
    plt.show()


def main():
    # q4(10, 2000)
    q6(10,2000,1000)
    # q7(10,2000,1000)


if __name__ == "__main__":
    main()
