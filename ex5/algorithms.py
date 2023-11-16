import gym
from typing import Optional, Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


# def generate_episode_custom(env: gym.Env, policy: Callable, es: bool = False):
#     """A function to generate one episode and collect the sequence of (s, a, r) tuples

#     This function will be useful for implementing the MC methods

#     Args:
#         env (gym.Env): a Gym API compatible environment
#         policy (Callable): A function that represents the policy.
#         es (bool): Whether to use exploring starts or not
#     """
#     episode = []
#     state = env.reset()["agent"]
#     while True:
#         # print(episode)
#         if es and len(episode) == 0:
#             action = env.action_space.sample()
#         else:
#             action = policy(state)

#         next_state, reward, done, truncated, _ = env.step(action)
#         next_pos = next_state["agent"]
#         episode.append((state, action, reward))
#         if done or truncated:
#             break
#         state = next_pos

#     return episode

def create_epsilon_policy(Q: defaultdict, epsilon: float, return_p = False) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.choice(num_actions)
            policy = np.array(range(num_actions))
        else:
            policy = np.where(Q[state] == Q[state].max())[0]
            action = np.random.choice(np.where(Q[state] == Q[state].max())[0])

        if return_p:
            return action, policy
        else: return action

    return get_action


def obtain_fixed_policy(env: gym.Env, methods):
    S_dims = (10, 7)
    # Obtain fixed near-optimal policy
    _, Q = methods  # qlearning can also be used
    num_actions = len(Q[0])
    policy = create_epsilon_policy(Q, epsilon = 0.1, return_p= True)
    Policy = defaultdict(lambda: np.zeros(env.action_space.n))

    for x in range(S_dims[0]):
        for y in range(S_dims[1]):
            _, Policy[(x,y)] = policy((x,y))

    return Policy


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_steps: int, gamma: float, epsilon: float
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

    # returns = np.zeros(num_episodes)
    num_episode = np.zeros(num_steps)

    episode = []
    state = env.reset()["agent"]

    for i in trange(num_steps, desc="Num_steps", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        # episode = generate_episode_custom(env, policy)
        if len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, truncated, _ = env.step(action)
        next_pos = next_state["agent"]
        episode.append((state, action, reward))
        G = 0
        state = next_pos

        if done:
            state = env.reset()["agent"]
            num_episode[i] = num_episode[i - 1] + 1

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + gamma * G
                N[state][action] += 1
                Q[state][action] = Q[state][action] + (G - Q[state][action]) / N[state][action]
        else: num_episode[i] = num_episode[i - 1]

    return num_episode

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q,epsilon)

    # returns = np.zeros(num_episodes)
    num_episode = np.zeros(num_steps)
    # episode = []
    state = env.reset()["agent"]
    action = policy(state)

    for i in trange(num_steps, desc="Num_steps", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        next_state, reward, done, truncated, _ = env.step(action)
        next_pos = next_state["agent"]
        # episode.append((state, action, reward))

        next_action = policy(next_pos)
        Q[state][action] += step_size * (reward + gamma * Q[next_pos][next_action] - Q[state][action])
        
        state = next_pos
        action = next_action

        if done:
            state = env.reset()["agent"]
            action = policy(state)
            num_episode[i] = num_episode[i - 1] + 1
        else: num_episode[i] = num_episode[i - 1]

    
    return num_episode, Q


def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    n_step: int
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon)
    num_episode = np.zeros(num_steps)
    step_count = 0
    
    while step_count < num_steps:
        state = env.reset()["agent"]
        action = policy(state)
        t = 0
        T = float('inf')
        rewards = []
        states = [state]
        actions = [action]

        while True:
            if t < T:
                next_state, reward, done, truncated, _ = env.step(actions[t])
                next_pos = next_state["agent"]
                rewards.append(reward)
                states.append(next_pos)

                if done:
                    T = t + 1
                    num_episode[step_count] = num_episode[step_count - 1] + 1
                else:
                    num_episode[step_count] = num_episode[step_count - 1]
                    next_action = policy(next_pos)
                    actions.append(next_action)
                
                step_count += 1
                if step_count >= num_steps:
                    break
            
            tau = t - n_step + 1
            if tau >= 0:
                n_step_return = .0
                for i in range(tau + 1, min(tau + n_step, T)):
                    n_step_return += gamma ** (i - tau - 1) * rewards[i]
                
                if tau + n_step < T:
                    n_step_return += (gamma ** (n_step - 1)) * Q[states[min(tau + n_step, T) - 1]][actions[min(tau + n_step, T) - 1]]
                
                Q[states[tau]][actions[tau]] += step_size * (n_step_return - Q[states[tau]][actions[tau]])

            if tau == T - 1:
                break

            t += 1
    
    return num_episode

def nstep_td(
    env: gym.Env,
    episodes: int,
    gamma: float,
    step_size: float,
    n_step: int
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    step_size = 0.5
    policy = obtain_fixed_policy(env, sarsa(env, 100000, 1, 0.1, step_size))
    # a = np.random.choice(len(A), 1, p = Policy[s])[0]

    V = defaultdict(float)    # default value 0 for all states
    
    # for _ in trange(episodes, desc="Num_Episode", leave=False):
    #     state = env.reset()["agent"]
    #     while True:
    #         action = np.random.choice(Policy[state])
    #         S_, reward, done, truncated, _ = env.step(action)
    #         next_state = S_["agent"]
    #         V[state] += step_size * (reward + gamma * V[next_state] - V[state])
    #         state = next_state
    #         if done: break

    
    for _ in trange(episodes, desc="Num_Episode", leave=False):
        state = env.reset()["agent"]
        # action = policy(state)
        t = 0
        T = float('inf')
        rewards = []
        states = [state]
        # actions = [action]

        while True:
            if t < T:
                action = np.random.choice(policy[states[t]])
                next_state, reward, done, truncated, _ = env.step(action)
                next_pos = next_state["agent"]
                rewards.append(reward)
                states.append(next_pos)

                if done:
                    T = t + 1
            
            tau = t - n_step + 1
            if tau >= 0:
                G = .0
                for i in range(tau + 1, min(tau + n_step, T)):
                    G += gamma ** (i - tau - 1) * rewards[i]
                
                if tau + n_step < T:
                    G += (gamma ** (n_step - 1)) * V[states[min(tau + n_step, T) - 1]]
                
                V[states[tau]] += step_size * (G - V[states[tau]])

            if tau == T - 1:
                break

            t += 1
    
    return V, policy

def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon)
    num_actions = 4

    # returns = np.zeros(num_episodes)
    num_episode = np.zeros(num_steps)
    # episode = []
    state = env.reset()["agent"]
    action = policy(state)

    for i in trange(num_steps, desc="Num_steps", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        next_state, reward, done, truncated, _ = env.step(action)
        next_pos = next_state["agent"]
        # episode.append((state, action, reward))

        next_action = policy(next_pos)

        expected_next_value = epsilon / num_actions * np.sum(Q[next_pos]) + (1 - epsilon) * np.max(Q[next_pos])
        Q[state][action] += step_size * (reward + gamma * expected_next_value - Q[state][action])
    
        state = next_pos
        action = next_action

        if done:
            state = env.reset()["agent"]
            action = policy(state)
            num_episode[i] = num_episode[i - 1] + 1
        else: num_episode[i] = num_episode[i - 1]

    
    return num_episode


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q,epsilon)

    # returns = np.zeros(num_episodes)
    num_episode = np.zeros(num_steps)
    state = env.reset()["agent"]

    for i in trange(num_steps, desc="Num_steps", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        action = policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_pos = next_state["agent"]

        # next_action = policy(next_pos)
        Q[state][action] += step_size * (reward + gamma * Q[next_pos].max() - Q[state][action])
        
        state = next_pos

        if done:
            state = env.reset()["agent"]
            num_episode[i] = num_episode[i - 1] + 1
        else: num_episode[i] = num_episode[i - 1]

    
    return num_episode


def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    step_size = 0.5
    Policy = obtain_fixed_policy(env, sarsa(env, 100000, 1, 0.1, step_size))
    # a = np.random.choice(len(A), 1, p = Policy[s])[0]

    V = defaultdict(float)    # default value 0 for all states
    
    for _ in trange(episodes, desc="Num_Episode", leave=False):
        state = env.reset()["agent"]
        while True:
            action = np.random.choice(Policy[state])
            S_, reward, done, truncated, _ = env.step(action)
            next_state = S_["agent"]
            V[state] += step_size * (reward + gamma * V[next_state] - V[state])
            state = next_state
            if done: break
                
    return V, Policy

def generate_episode(env: gym.Env, policy, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()["agent"]
    while True:
        action = np.random.choice(policy[state])

        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state['agent']

    return episode

def on_policy_mc_evaluation(
    env: gym.Env,
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
    step_size = 0.5
    policy = obtain_fixed_policy(env, sarsa(env, 100000, 1, 0.1, step_size))

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
    return V, policy

def tdEvl(epsLs, V):
    targetLs = []
    for i in range(len(epsLs)):
        eps = epsLs[i]
        for n in range(len(eps)-1):
            state = eps[n][0]
            if state == (0,3):
                nxtState = eps[n+1][0]
                target = -1 + V[nxtState]
                targetLs.append(target)
                
    return targetLs

def mcEvl(epsLs):
        targetLs = []
        for i in range(len(epsLs)):
            eps = epsLs[i]
            targetLs.append(-len(eps)-1)
                    
        return targetLs

def ntdEvl(epsLs, V, nStep = 4):
    targetLs = []
    for i in range(len(epsLs)):
        eps = epsLs[i]
        for n in range(len(eps) + nStep-1):
            tao = n - nStep + 1
            if tao == 0:
                h = min(len(eps)-1, nStep + tao)    
                nxtState = eps[h][0]
                
                g = -(h-tao)
                if tao + nStep < len(eps):
                    g += V[nxtState[0], nxtState[1]]
                targetLs.append(g)
    
    return targetLs

def learning_targets(
    env: gym.Env, V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    # targets = np.zeros(len(episodes))
    # a = algorithms.td_prediction(Windy_env, 1, 10)
    # a = algorithms.nstep_td(Windy_env, 10, 1, 0.5, 4)
    # a = algorithms.on_policy_mc_evaluation(Windy_env, 10, 1)

    # N = defaultdict(int)
    # step_size = 0.5
    # episodes_ls = []
    # trueVal =  V[3,0]
    # policy = obtain_fixed_policy(env, sarsa(env, 10000, 1, 0.1, step_size))

    # for _ in trange(episodes, desc="Episode"):
    #     episodes_ls.append(generate_episode(env, policy))
    
    if not n:
        tgtLs = mcEvl(episodes)
        # returns = 0
        # for t in range(len(episodes) - 1, -1, -1):
        #     _, _, reward = episodes[t][0]
        #     returns += reward
        #     targets[t] = returns
    elif n==1:
        tgtLs = tdEvl(episodes, V)
    else: tgtLs = ntdEvl(episodes, V)


    # fig, ax = plt.subplots()
    plt.figure()
    plt.hist(tgtLs)
    # ax.set_xlabel('G')
    # ax.set_title('Monte-Carlo: N=1 (converged)')
    # ax.vlines(trueVal,0,110, color = 'black', linestyles = 'dashed')
    plt.show()

        
