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
        self._place_obstacles()
        # ✅ make obstacles accessible to rendering.py
        self.env.obstacles = self.obstacles
        return obs, info

    def _place_obstacles(self):
        self.obstacles = set()
        height, width = self.env.field_size
        while len(self.obstacles) < self.num_obstacles:
            x = self.rng.randint(0, height - 1)
            y = self.rng.randint(0, width - 1)
            if self.env.field[x, y] == 0:  # only empty cells
                self.obstacles.add((x, y))

    def step(self, actions):
        MOVES = {
            0: (0, 0),   # no-op
            1: (-1, 0),  # up
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 1),   # right
            5: (0, 0),   # load food
        }
        new_actions = []
        for i, action in enumerate(actions):
            dx, dy = MOVES.get(action, (0, 0))
            pos = np.argwhere(self.env.field == i + 1)
            if len(pos) > 0:
                x, y = pos[0]
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.obstacles:
                    action = 0  # block into obstacle
            new_actions.append(action)
        return self.env.step(new_actions)



if __name__ == "__main__":
    env = ForagingEnv(
        players=3,
        min_player_level=1,
        max_player_level=3,
        field_size=(10, 10),
        min_food_level=1,
        max_food_level=3,
        max_num_food=4,
        sight=3,
        max_episode_steps=50,
        force_coop=False,
        render_mode="ansi",  # or "human" or "rgb_array"
    )

    env = ForagingObstacleWrapper(env, num_obstacles=10, seed=42)
    obs, info = env.reset(seed=0)

    #print("Initial obstacles:", env.obstacles)
    print(env.render())  # see board with X as obstacles
    time.sleep(3)
