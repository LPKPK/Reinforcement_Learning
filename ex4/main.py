import gym
import policy
import algorithms
import plot
import env
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def Q3_a():
    V_500k = algorithms.on_policy_mc_evaluation(gym.make("Blackjack-v1"), policy.default_blackjack_policy, 500000, 1)

    Used_A = np.zeros([21, 10])
    notUsed_A = np.zeros([21, 10])

    for i in V_500k.keys():
        player, dealer, ace = i
        if ace:
            Used_A[player - 1, dealer - 1] = V_500k[i]
        else:
            notUsed_A[player - 1, dealer - 1] = V_500k[i]
    
    plot.Q3_plot_value(Used_A, "Usable Ace After 500k")
    plot.Q3_plot_value(notUsed_A, "Non-usable Ace After 500k")

    V_10k = algorithms.on_policy_mc_evaluation(gym.make("Blackjack-v1"), policy.default_blackjack_policy, 10000, 1)
    
    Used_A = np.zeros([21, 10])
    notUsed_A = np.zeros([21, 10])

    for i in V_10k.keys():
        player, dealer, ace = i
        if ace:
            Used_A[player - 1, dealer - 1] = V_10k[i]
        else:
            notUsed_A[player - 1, dealer - 1] = V_10k[i]

    plot.Q3_plot_value(Used_A, "Usable Ace After 10k")
    plot.Q3_plot_value(notUsed_A, "Non-usable Ace After 10k")
    plt.show()

def Q3_b():
    Q, policy = algorithms.on_policy_mc_control_es(gym.make("Blackjack-v1"), 5000000, 1)
    
    Used_A = np.zeros([21, 10])
    notUsed_A = np.zeros([21, 10])
    UA_Policy = np.zeros([21, 10])
    nUA_Policy = np.zeros([21, 10])

    for i in Q.keys():
        player, dealer, ace = i
        if ace:
            Used_A[player - 1, dealer - 1] = np.max(Q[i])
            UA_Policy[player - 1, dealer - 1] = np.argmax(Q[i])

        else:
            notUsed_A[player - 1, dealer - 1] = np.max(Q[i])
            nUA_Policy[player - 1, dealer - 1] = np.argmax(Q[i])

    
    plot.Q3_plot_policy(UA_Policy, "The Optimal Policy with usable Ace")
    plot.Q3_plot_policy(nUA_Policy, "The Optimal Policy with non-usable Ace")
    plot.Q3_plot_value(Used_A, "State-value with Usable Ace")
    plot.Q3_plot_value(notUsed_A, "State-value with Non-usable Ace")
    plt.show()

def Q4_a():
    env.register_env()
    num_episodes = 1000
    trials = 1

    # state, info= FRoom_env.reset()
    # for i in range(5):
    returns = algorithms.on_policy_mc_control_epsilon_soft(gym.make('FourRooms-v0'), num_episodes, 0.99, 0.1)
    
    plt.figure()
    plt.plot(returns)
    plt.show()



def Q4_b():
    env.register_env()
    num_episodes = 10000
    trials = 10
    epsilon = [0.1, 0.02, 0]
    colors = ['tab:blue', 'tab:green', 'tab:red']
    plt.figure()


    for i in range(len(epsilon)):
        return_t = []
        for t in range(trials):
            returns = algorithms.on_policy_mc_control_epsilon_soft(gym.make('FourRooms-v0'), num_episodes, 0.99, epsilon[i])
            return_t.append(returns)

        err = 1.96 * np.std(returns, 0) / np.sqrt(trials)
        return_avr = np.average(return_t, 0)
        # print(err)
        # print(return_avr)
        plt.plot(return_avr, color = colors[i])
        plt.fill_between(np.arange(num_episodes), return_avr + err, return_avr - err, alpha = 0.3, color = colors[i])
    plt.show()



if __name__ == "__main__":
    Q4_b()