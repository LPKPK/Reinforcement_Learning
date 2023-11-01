import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()[0]
    while True:
        # print(episode)
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode

def generate_episode_custom(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()[0]["agent"]
    while True:
        # print(episode)
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _, _ = env.step(action)
        next_pos = next_state["agent"]
        episode.append((state, action, reward))
        if done:
            break
        state = next_pos

    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    policy: Callable,
    num_episodes: int,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy)

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            # TODO Q3a
            # Update V and N here according to first visit MC
            state, action, reward = episode[t]
            N[state] += 1   # First-Visit
            G = reward + gamma * G
            V[state] +=  (G - V[state]) / N[state]
    return V


def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float
) -> Tuple[defaultdict, Callable]:
    """On-policy Monte Carlo control with exploring starts for Blackjack

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_blackjack_policy(Q)

    for _ in trange(num_episodes, desc="Episode"):
        # TODO Q3b
        episode = generate_episode(env, policy, True)
        G = 0

        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G
            N[state][action] += 1
            Q[state][action] = Q[state][action] + (G - Q[state][action]) / N[state][action]
        
    return Q, policy


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)

    returns = np.zeros(num_episodes)

    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode_custom(env, policy)
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G
            N[state][action] += 1
            Q[state][action] = Q[state][action] + (G - Q[state][action]) / N[state][action]

        returns[i] = G

    return returns
