from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple
import matplotlib.pyplot as plt


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (0, -1),
        Action.DOWN: (1, 0),
        Action.RIGHT: (0, 1),
        Action.UP: (-1, 0),
    }
    return mapping[action]


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # set the locations of A and B, the next locations, and their rewards
        self.A = (0,1)
        self.A_prime = (4,1)
        self.A_reward = 10
        self.B = (0,3)
        self.B_prime = (2,3)
        self.B_reward = 5

    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """

        next_state = None
        reward = None
        action_taken = actions_to_dxdy(action)

        # Check if current state is A and B and return the next state and corresponding reward
        # Else, check if the next step is within boundaries and return next state and reward
        if state == self.A:
            next_state = self.A_prime
            reward = self.A_reward
        elif state == self.B:
            next_state = self.B_prime
            reward = self.B_reward
        else:
            next_state = tuple(np.add(state, action_taken))
            if next_state in self.state_space:
                reward = 0
            else:
                next_state = state
                reward = -1

        return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        next_state, reward = self.transitions(state, action)
        # compute the expected return
        ret = reward + gamma * V[next_state]

        return ret
    
    def Q5_a(self):
        V_s = np.zeros((5, 5), dtype=float)
        theta = 0.001
        gamma = 0.9
        while True:
            delta = 0
            for s in self.state_space:
                v = 0
                action_prob = 1/4   #   equiprobable random policy
                for a in range(self.action_space):
                    v += action_prob * self.expected_return(V_s, s, a, gamma)
                delta = max(delta, np.abs(V_s[s] - v))
                V_s[s] = v
            if delta < theta:
                break
        return V_s
    
    def Q5_b(self):
        V_s = np.zeros((5, 5), dtype=float)
        theta = 0.001
        gamma = 0.9
        pi = {   
            0 : "LEFT",
            1 : "DOWN",
            2 : "RIGHT",
            3 : "UP" }
        row = []
        pi_s = []

        while True:
            delta = 0
            for s in self.state_space:
                temp = []
                for a in range(self.action_space):
                    temp.append(self.expected_return(V_s, s, a, gamma))
                delta = max(delta, np.abs(V_s[s] - max(temp)))
                V_s[s] = max(temp)
            if delta < theta:
                break
        # print Policy
        for s in self.state_space:
            temp = []
            element = []
            for a in range(self.action_space):
                temp.append(self.expected_return(V_s, s, a, gamma))
            for i in np.where(temp == max(temp))[0]:
                element.append(pi[i])
            row.append(element)
            if s[1] == self.cols-1:
                pi_s.append(row)
                row = []
                  
        return V_s, pi_s
    
    def Q5_c(self):
        V_s = np.zeros((5, 5), dtype=float)
        theta = 0.001
        gamma = 0.9
        pi = {   
            0 : "LEFT",
            1 : "DOWN",
            2 : "RIGHT",
            3 : "UP" }
        row = []
        pi_s = []

        # initialized policy
        policy = [[[np.random.randint(0, 3)] for _ in range(5)] for _ in range(5)]
        policy_stable = False

        while not policy_stable:
            # Policy Evaluation
            while True:
                delta = 0
                for s in self.state_space:
                    # 
                    v = V_s[s]
                    V_s[s] = self.expected_return(V_s, s, policy[s[0]][s[1]][0], gamma)
                    delta = max(delta, np.abs(V_s[s] - v))
                if delta < theta:
                    break
            # Policy Iteration
            policy_stable = True
            for s in self.state_space:
                old_act = policy[s[0]][s[1]][0]
                policy[s[0]][s[1]] = []
                temp = []

                for a in range(self.action_space):
                    temp.append(self.expected_return(V_s, s, a, gamma))                
                
                policy[s[0]][s[1]] = np.where(temp == max(temp))[0]
                best_act = np.random.choice(policy[s[0]][s[1]])

                # Fix in Q2(a)
                best_action_value = self.expected_return(V_s, s, best_act, gamma)
                old_action_value = self.expected_return(V_s, s, old_act, gamma)
                if (old_action_value != best_action_value):
                    policy_stable = False
        # Print Policy
        for s in self.state_space:
            element = []
            for i in policy[s[0]][s[1]]:
                element.append(pi[i])
            policy[s[0]][s[1]] = element

        return V_s, policy



class JacksCarRental:
    def __init__(self, modified: bool = False) -> None:
        """JacksCarRental

        Args:
           modified (bool): False = original problem Q6a, True = modified problem for Q6b

        State: tuple of (# cars at location A, # cars at location B)

        Action (int): -5 to +5
            Positive if moving cars from location A to B
            Negative if moving cars from location B to A
        """
        self.modified = modified

        self.action_space = list(range(-5, 6))

        self.rent_reward = 10
        self.move_cost = 2

        # For modified problem
        self.overflow_cars = 10
        self.overflow_cost = 4

        # Rent and return Poisson process parameters
        # Save as an array for each location (Loc A, Loc B)
        self.rent = [poisson(3), poisson(4)]
        self.return_ = [poisson(3), poisson(2)]

        # Max number of cars at end of day
        self.max_cars_end = 20
        # Max number of cars at start of day
        self.max_cars_start = self.max_cars_end + max(self.action_space)

        self.state_space = [
            (x, y)
            for x in range(0, self.max_cars_end + 1)
            for y in range(0, self.max_cars_end + 1)
        ]

        # Store all possible transitions here as a multi-dimensional array (locA, locB, action, locA', locB')
        # This is the 3-argument transition function p(s'|s,a)
        self.t = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                len(self.action_space),
                self.max_cars_end + 1,
                self.max_cars_end + 1,
            ),
        )

        # Store all possible rewards (locA, locB, action)
        # This is the reward function r(s,a)
        self.r = np.zeros(
            (self.max_cars_end + 1, self.max_cars_end + 1, len(self.action_space))
        )
        self.precompute_transitions()

    def _open_to_close(self, loc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the probability of ending the day with s_end \in [0,20] cars given that the location started with s_start \in [0, 20+5] cars.

        Args:
            loc_idx (int): the location index. 0 is for A and 1 is for B. All other values are invalid
        Returns:
            probs (np.ndarray): list of probabilities for all possible combination of s_start and s_end
            rewards (np.ndarray): average rewards for all possible s_start
        """
        probs = np.zeros((self.max_cars_start + 1, 31))
        # real_probs = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        rewards = np.zeros(self.max_cars_start + 1)

        # p_rent, p_return = self.make_likelihood(loc_idx)

        for start in range(probs.shape[0]):
            # TODO Calculate average rewards.
            # For all possible s_start, calculate the probability of renting k cars.
            # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
            
            avg_rent = 0.0
            for rent in range(start):  # renting should smaller than start
                avg_rent += rent * self.rent[loc_idx].pmf(rent)

            avg_rent += start * (1 - self.rent[loc_idx].cdf(start - 1))
            rewards[start] = self.rent_reward * avg_rent

            # TODO Calculate probabilities
            # Loop over every possible s_end
            for end in range(probs.shape[1]):
                prob = 0.0
                # Since s_start and s_end are specified,
                # you must rent a minimum of max(0, start-end)
                min_rent = max(0, start - end)

                # TODO Loop over all possible rent scenarios and compute probabilities
                # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
                for i in range(min_rent, start):
                    # if (start - end) >= 0:
                    # prob += p_rent[start][i] * p_return[end][i - (start - end)]
                    prob += self.rent[loc_idx].pmf(i) * self.return_[loc_idx].pmf(i - (start - end))


                # prob += p_rent[start][start] * self.return_[loc_idx].pmf(end)
                # if end != (probs.shape[1] - 1):
                prob += (1 - self.rent[loc_idx].cdf(start - 1)) * self.return_[loc_idx].pmf(end)
                # else: prob += (1 - self.rent[loc_idx].cdf(start - 1)) * (1 - self.rent[loc_idx].cdf(end - 1))
                probs[start, end] = prob
        
        sum_last_three = np.sum(probs[:, -10:], axis=1)
        new_data = probs[:, :-10]
        new_data = np.column_stack((new_data, sum_last_three))
        # real_probs = [col[:20] + [sum(col[20:26])] for col in probs]

        # probs = self.make_likelihood(loc_idx)

        return new_data, rewards

    # Reward as a function of (N1'', N2'')
    # Total reward is R_X + R_A * |a|
    def right_tail(self, p):
        """
        Adjust the last value to account for an overflow over N_MAX_CARS
        """
        p[-1] = 1 - np.sum(p[:-1])
        return p
    

    def make_likelihood(self, loc_idx):
        """
        Create P(Ni' | Ni")
        """        
        rent = np.arange(self.max_cars_start + 1)
        retu = np.arange(self.max_cars_end + 1)
        p = self.rent[loc_idx].pmf(rent)
        px = []   
        for i in rent: px.append(self.right_tail(p[:i+1].copy()))
        
        p = self.return_[loc_idx].pmf(retu)
        py = []
        for i in retu: py.append(self.right_tail(p[:i+1].copy()))
        
        # for N__ in range(self.max_cars_end + 1):
        #     px_N__ = px[N__]

        #     for N_ in range(self.max_cars_start + 1):     
        #         for x in range(N__ + 1):
        #             py_N_ = py[self.max_cars_start - N__ + x]

        #             y = N_ - N__ + x
        #             if y < 0: continue

        #             P[N_, N__] += px_N__[x] * py_N_[y]
                
        return px, py


    
    def _calculate_cost(self, state: Tuple[int, int], action: int) -> float:
        """A helper function to compute the cost of moving cars for a given (state, action)

        Note that you should compute costs differently if this is the modified problem.

        Args:
            state (Tuple[int,int]): state
            action (int): action
        """
        # Q6 (a)
        cost = self.move_cost * abs(action)

        # # Q6 (b)
        # locA_, locB_ = state
        # if action > 0:
        #     cost = self.move_cost * (abs(action)-1)
        # else: cost = self.move_cost * abs(action)

        # if locA_ - action > 10:
        #     cost += 4
        # if locB_ + action > 10:
        #     cost += 4

        return cost

    def _valid_action(self, state: Tuple[int, int], action: int) -> bool:
        """Helper function to check if this action is valid for the given state

        Args:
            state:
            action:
        """
        if state[0] < action or state[1] < -(action):
            return False
        else:
            return True

    def precompute_transitions(self) -> None:
        """Function to precompute the transitions and rewards.

        This function should have been run at least once before calling expected_return().
        You can call this function in __init__() or separately.

        """
        # Calculate open_to_close for each location
        day_probs_A, day_rewards_A = self._open_to_close(0)
        day_probs_B, day_rewards_B = self._open_to_close(1)

        # Perform action first then calculate daytime probabilities
        for locA in range(self.max_cars_end + 1):
            for locB in range(self.max_cars_end + 1):
                for ia, action in enumerate(self.action_space):
                    # Check boundary conditions
                    if not self._valid_action((locA, locB), action):
                        self.t[locA, locB, ia, :, :] = 0
                        self.r[locA, locB, ia] = 0
                    else:
                        # TODO Calculate day rewards from renting
                        # Use day_rewards_A and day_rewards_B and _calculate_cost()
                        self.r[locA, locB, ia] = day_rewards_A[locA - action] + day_rewards_B[locB + action] - self._calculate_cost((locA,locB), action)

                        # Loop over all combinations of locA_ and locB_
                        for locA_ in range(self.max_cars_end + 1):
                            for locB_ in range(self.max_cars_end + 1):

                                # TODO Calculate transition probabilities
                                # Use the probabilities computed from open_to_close
                                self.t[locA, locB, ia, locA_, locB_] = day_probs_A[locA - action][locA_] * day_probs_B[locB + action][locB_]

    def transitions(self, state: Tuple, action: Action) -> np.ndarray:
        """Get transition probabilities for given (state, action) pair.

        Note that this is the 3-argument transition version p(s'|s,a).
        This particular environment has stochastic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            probs (np.ndarray): return probabilities for next states. Since transition function is of shape (locA, locB, action, locA', locB'), probs should be of shape (locA', locB')
        """
        # TODO

        ia = action + 5
        locA, locB = state
        probs = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                ))

        for locA_ in range(self.max_cars_end + 1):
            for locB_ in range(self.max_cars_end + 1):
                probs[locA_, locB_] = self.t[locA, locB, ia, locA_, locB_]

        return probs

    def rewards(self, state, action) -> float:
        """Reward function r(s,a)

        Args:
            state (Tuple): state
            action (Action): action
        Returns:
            reward: float
        """
        # TODO
        ia = action + 5
        locA, locB = state
        reward = self.r[locA, locB, ia]

        return reward


    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        # TODO compute the expected return
        reward = self.rewards(state, action)
        probs = self.transitions(state, action)

        # compute the expected return
        ret = reward + np.sum(probs * gamma * V)
        return ret

    def Q6(self):
        V_s = np.zeros((self.max_cars_end + 1, self.max_cars_end + 1))
        theta = 0.001
        gamma = 0.9

        # initialized policy
        policy = [[[np.random.randint(-5, 5)] for _ in range(self.max_cars_end + 1)] for _ in range(self.max_cars_end + 1)]
        policy_stable = False
        iteration = 0

        while not policy_stable:
            # Policy Evaluation
            print("Policy Iteration ", iteration)
            while True:
                delta = 0
                for s in self.state_space:
                    # 
                    v = V_s[s]
                    V_s[s] = self.expected_return(V_s, s, policy[s[0]][s[1]][0], gamma)
                    delta = max(delta, np.abs(V_s[s] - v))
                if delta < theta:
                    break
            # Policy Iteration
            policy_stable = True
            for s in self.state_space:
                old_act = policy[s[0]][s[1]][0]
                policy[s[0]][s[1]] = []
                temp = []

                for a in self.action_space:
                    temp.append(self.expected_return(V_s, s, a, gamma))              
                
                policy[s[0]][s[1]] = np.where(temp == max(temp))[0] - 5
                best_act = np.random.choice(policy[s[0]][s[1]])

                # Fix in Q2(a)
                best_action_value = self.expected_return(V_s, s, best_act, gamma)
                old_action_value = self.expected_return(V_s, s, old_act, gamma)
                if (old_action_value != best_action_value):
                    policy_stable = False

            # Print Policy
            plt.figure(figsize=(12, 10))
            plot = plt.imshow(policy, cmap="RdGy")
            plt.gca().invert_yaxis()

            bounds = np.arange(-5, 6)

            plt.colorbar(plot, boundaries=bounds, ticks=bounds)
            plt.title(f"Policy at iteration {iteration}")
            plt.xlabel("Number of cars at LocB")
            plt.ylabel("Number of cars at LocA")
            plt.yticks(np.arange(0, self.max_cars_end + 1, 5))
            plt.xticks(np.arange(0, self.max_cars_end + 1, 5))
            
            iteration += 1

        # Show Value
        x = np.linspace(0, 20, 21)
        y = np.linspace(0, 20, 21)

        X, Y = np.meshgrid(x, y)
        Z = V_s
        
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('State Value')

        plt.show()

        return V_s, policy

