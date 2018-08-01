from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class VICEnvRandom(MiniGridEnv):
    """
    Environment with upto 11x11 dimensions 
    """

    def __init__(self, mode='rgb', action_mode='forward'):

        super(VICEnvRandom, self).__init__(
            grid_size=11,
            max_steps=20000,
            see_through_walls=True,
            mode=mode,
            action_mode=action_mode
        )

    def _gen_grid(self, width, height, start_pos=None, start_dir=None):
        
        width = random.randint(4, 11)
        height = random.randint(4, 11)
        self.grid = Grid(11, 11)
        wall = Wall()

        # Draw the left, center-column and right walls
        for i in range(0, height):
            self.grid.set(0, i, wall)
            self.grid.set(width-1, i, wall)
        # Draw the top, center-row and bottom walls
        for i in range(0, width):
            self.grid.set(i, 0, wall)
            self.grid.set(i, height-1, wall) 
        
        self.valid_starts = [(x, y) for x in range(1,width-1) for y in range(1, height-1)]

        self.start_pos = random.sample(self.valid_starts, 1)[0]

        self.start_dir = 0

        self.mission = 'just roam around'

class VICGridEnvRandom(VICEnvRandom):
    def __init__(self):
        super().__init__(mode='grid')

class VICEnvOmniRandom(VICEnvRandom):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='omni')

class VICGridEnvOmniRandom(VICEnvRandom):
    def __init__(self):
        super().__init__(mode='grid', action_mode='omni')

register(
    id='MiniGrid-VICEnvRandom-v0',
    entry_point='gym_minigrid.envs:VICEnvRandom',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnvRandom-v1',
    entry_point='gym_minigrid.envs:VICGridEnvRandom',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnvRandom-v2',
    entry_point='gym_minigrid.envs:VICEnvOmniRandom',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnvRandom-v3',
    entry_point='gym_minigrid.envs:VICGridEnvOmniRandom',
    reward_threshold=1000.0
)

