import gymnasium as gym
import numpy as np
import random
import time

from lbforaging.foraging.environment import ForagingEnv


class ForagingObstacleWrapper(gym.Wrapper):
    def __init__(self, env, num_obstacles=5, seed=None):
        super().__init__(env)
        self.num_obstacles = num_obstacles
        self.rng = random.Random(seed)
        self.obstacles = set()

    def reset(self, **kwargs):
        
        obs, info = self.env.reset(**kwargs)
        #print(self.env.field)
        num_players = len(self.env.players)

    # foods
        num_foods = np.count_nonzero(self.env.field > 0)

        self._place_obstacles()
        if num_foods<4:
            print(f"[DEBUG] Players: {num_players}, Food items: {num_foods}----------------------------------------")

        current = self.env
        while hasattr(current, "env"):
            setattr(current, "obstacles", self.obstacles)
            current = current.env
        setattr(current, "obstacles", self.obstacles)

        obs = self._add_obstacles_to_obs(obs) #under testing

        return obs, info


    def _place_obstacles(self):
        #self.obstacles = {
           # (1, 7), (3, 8), (6, 7), (8, 6),
            #(2, 4), (1, 3), (3, 2), (5, 1),
           # (7, 2), (9, 4)
        #}

    
    
    
    
        self.obstacles = set()
        height, width = self.env.field_size
        while len(self.obstacles) < self.num_obstacles:
            x = self.rng.randint(0, height - 1)
            y = self.rng.randint(0, width - 1)

            # ✅ skip if food is here
            if self.env.field[x, y] != 0:
                continue

            # ✅ skip if any agent is here
            if any(p.position == (x, y) for p in self.env.players):
               continue

            self.obstacles.add((x, y))
        #print("Obstacles placed at:", sorted(self.obstacles))


    def step(self, actions):
        MOVES = {
            0: (0, 0),   # no-op
            1: (-1, 0),  # up
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 1),   # right
            5: (0, 0),   # load food
        }
        height, width = self.env.field_size
        new_actions = []
        
        for player, action in zip(self.env.players, actions):
            dx, dy = MOVES.get(action, (0, 0))
            x, y = player.position
            nx, ny = x + dx, y + dy

            # boundary check
            if not (0 <= nx < height and 0 <= ny < width):
                action = 0
            # obstacle check
            elif (nx, ny) in self.obstacles:
                action = 0

            new_actions.append(action)

        return self.env.step(new_actions)
    
    def _add_obstacles_to_obs(self, obs):
        """
        Add an extra layer for obstacles:
        - If obs is a grid: append a channel with 1 where obstacles are, else 0.
        - If obs is a vector: you could flatten and append obstacle coords.
        """
        new_obs = []
        for o in obs:  # each agent’s observation
            if isinstance(o, np.ndarray) and len(o.shape) == 3:
                # grid obs: (layers, H, W)
                obstacle_layer = np.zeros_like(o[0])
                for (x, y) in self.obstacles:
                    obstacle_layer[x, y] = 1
                o = np.concatenate([o, obstacle_layer[None, :, :]], axis=0)
            new_obs.append(o)
        return tuple(new_obs)



