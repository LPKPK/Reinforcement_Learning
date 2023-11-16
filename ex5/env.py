from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    # TODO
    register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")



class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    NO_MOVEMENT = 8

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
        Action.UP_LEFT: (-1, 1),
        Action.UP_RIGHT: (1, 1),
        Action.DOWN_LEFT: (-1, -1),
        Action.DOWN_RIGHT: (1, -2),
        Action.NO_MOVEMENT: (0, 0)
    }
    return mapping[action]


class WindyGridWorldEnv(Env):
    def __init__(self, king_mode = False, nine_action = False, stc_wind = False):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = np.zeros((self.rows, self.cols), dtype=int)
        for k in range(self.cols):
            for i in range(self.rows):
                if (3 <= i < 6) or i == 8:
                    self.wind[i][k] = 1
                elif 6 <= i <8:
                    self.wind[i][k] = 2

        if king_mode:
            if nine_action:
                self.action_space = spaces.Discrete(len(Action))
            else: self.action_space = spaces.Discrete(len(Action)-1)
        else: self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        self.truncated = False
        self.stc_wind = stc_wind
        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return {"agent": self.agent_pos, "goal": self.goal_pos}
    
    def _get_info(self):
        return None
    
    def reset(self):
        self.agent_pos = self.start_pos
        observation = self._get_obs()

        return observation

    def _check_finish(self):
        if self.agent_pos == self.goal_pos:
            return True
        return False
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # TODO
        action_taken = actions_to_dxdy(action)
        next_pos = np.add(self.agent_pos, action_taken)
        x,y = self.agent_pos
        # wind_index = np.dstack(np.where(self.wind != 0))
        # if next_pos in wind_index[0]:
        #     next_pos = (x, y + self.wind[x][y])

        bounds = (self.rows, self.cols)
        if all(0 <= a < b for a, b in zip(next_pos, bounds)):
            if not self.stc_wind:
                next_state = (next_pos[0], next_pos[1] + self.wind[x,y])
            else: 
                if 3 <= self.agent_pos[0] < 9:
                    wind_aff = np.random.choice([-1, 0, 1])
                    next_state = (next_pos[0], next_pos[1] + self.wind[x,y] + wind_aff)
                else: next_state = (next_pos[0], next_pos[1] + self.wind[x,y])

            if next_state[1] >= self.cols:
                next_state = next_pos
        else:
            next_state = self.agent_pos

        self.agent_pos = tuple(next_state)

        if self._check_finish():
            terminated = True
            reward = 0
        else:
            terminated = False
            reward = -1

        observation = self._get_obs()

        return observation, reward, terminated, self.truncated, {}
