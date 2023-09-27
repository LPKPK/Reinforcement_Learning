import matplotlib.pyplot as plt
from env import BanditEnv
from tqdm import trange


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
    # TODO initialize env and agents here
    env = None
    agents = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for agent in agents:
            agent.reset()

        # TODO For each trial, perform specified number of steps for each type of agent

    pass


def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = None
    agents = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for agent in agents:
            agent.reset()

        # TODO For each trial, perform specified number of steps for each type of agent

    pass


def main():
    q4(10, 2000)


if __name__ == "__main__":
    main()
