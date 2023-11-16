import env
import gym
import algorithms
import matplotlib.pyplot as plt
import numpy as np

    
def Q4_b_c_d(king_mode, nine_action, stochastic_wind):
    env.register_env()
    Windy_env = gym.make('WindyGridWorld-v0', king_mode = king_mode, nine_action = nine_action, stc_wind = stochastic_wind)

    # on_policy_mc = algorithms.on_policy_mc_control_epsilon_soft(Windy_env, 90000,  0.99, 0.1)
    # sarsa = algorithms.sarsa(Windy_env, 8000, 1, 0.1, 0.5)
    # exp_sarsa = algorithms.exp_sarsa(Windy_env, 8000, 1, 0.1, 0.5)
    # q_learn = algorithms.q_learning(Windy_env, 8000, 1, 0.1, 0.5)
    # n_step = algorithms.nstep_sarsa(Windy_env, 8000, 1, 0.1, 0.5, 4)

    trials = 10
    labels = ['on_policy_mc', 'sarsa', 'exp_sarsa', 'q_learn', 'n_step']
    colors = ['tab:brown', 'tab:green', 'tab:red', 'tab:orange', 'tab:blue']
    plt.figure()


    # for i in range(len(methods)):
    num_step = 10000
    return_t = []
    for t in range(trials):
        returns = algorithms.on_policy_mc_control_epsilon_soft(Windy_env, num_step,  1, 0.1)
        return_t.append(returns)

    err = 1.96 * np.std(return_t, 0) / np.sqrt(trials)
    return_avr = np.average(return_t, 0)
    plt.plot(return_avr)
    plt.plot(return_avr, label = labels[0], color = colors[0])
    plt.fill_between(np.arange(num_step), return_avr + err, return_avr - err, alpha = 0.2, color = colors[0])
        # plt.title(f"")
    
    return_t = []
    for t in range(trials):
        returns, Q = algorithms.sarsa(Windy_env, num_step, 1, 0.1, 0.5)
        return_t.append(returns)

    # print(Q)
    err = 1.96 * np.std(return_t, 0) / np.sqrt(trials)
    return_avr = np.average(return_t, 0)
    plt.plot(return_avr)
    plt.plot(return_avr, label = labels[1], color = colors[1])
    plt.fill_between(np.arange(num_step), return_avr + err, return_avr - err, alpha = 0.2, color = colors[1])

    return_t = []
    for t in range(trials):
        returns = algorithms.exp_sarsa(Windy_env, num_step, 1, 0.1, 0.5)
        return_t.append(returns)

    err = 1.96 * np.std(return_t, 0) / np.sqrt(trials)
    return_avr = np.average(return_t, 0)
    plt.plot(return_avr)
    plt.plot(return_avr, label = labels[2], color = colors[2])
    plt.fill_between(np.arange(num_step), return_avr + err, return_avr - err, alpha = 0.2, color = colors[2])

    return_t = []
    for t in range(trials):
        returns = algorithms.q_learning(Windy_env, num_step, 1, 0.1, 0.5)
        return_t.append(returns)

    err = 1.96 * np.std(return_t, 0) / np.sqrt(trials)
    return_avr = np.average(return_t, 0)
    plt.plot(return_avr)
    plt.plot(return_avr, label = labels[3], color = colors[3])
    plt.fill_between(np.arange(num_step), return_avr + err, return_avr - err, alpha = 0.2, color = colors[3])

    return_t = []
    n_step = 4
    for t in range(trials):
        returns = algorithms.nstep_sarsa(Windy_env, num_step, 1, 0.1, 0.5, n_step)
        return_t.append(returns)

    err = 1.96 * np.std(return_t, 0) / np.sqrt(trials)
    return_avr = np.average(return_t, 0)
    plt.plot(return_avr)
    plt.plot(return_avr, label = labels[4], color = colors[4])
    plt.fill_between(np.arange(num_step), return_avr + err, return_avr - err, alpha = 0.2, color = colors[4])


    plt.legend()
    plt.show()

def Q5(king_mode =False, nine_action = False, stochastic_wind = False):
    env.register_env()
    Windy_env = gym.make('WindyGridWorld-v0', king_mode = king_mode, nine_action = nine_action, stc_wind = stochastic_wind)

    episodes = []
    # V, policy = algorithms.td_prediction(Windy_env, 1, 10)
    V, policy = algorithms.nstep_td(Windy_env, 10, 1, 0.5, 4)
    # V, policy = algorithms.on_policy_mc_evaluation(Windy_env, 10, 1)

    # print(V)

    for _ in range(500):
        episodes.append(algorithms.generate_episode(Windy_env, policy))

    algorithms.learning_targets(Windy_env, V, 1, episodes, 4)

if __name__ == "__main__":
    """
    Q4b args: (king_mode = False, nine_action = False, stochastic_wind = False)
    Q4c args: (king_mode = True, nine_action = False(or True), stochastic_wind = False)
    Q4d args: (king_mode = True, nine_action = False, stochastic_wind = True)
    """
    Q4_b_c_d(king_mode = False, nine_action = False, stochastic_wind = False)

    # print(Q5())