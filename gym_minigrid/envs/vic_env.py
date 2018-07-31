from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class VICEnv(MiniGridEnv):
    """
    Environment with 10x6 dimensions (9x5 effective states) 
    """

    def __init__(self, mode='rgb', action_mode='forward', randomize=False):

        self.randomize = randomize
        self.valid_starts = [(x, y) for x in range(1,6) for y in range(1, 10)]
        super(VICEnv, self).__init__(
            grid_size=11,
            max_steps=20000,
            see_through_walls=True,
            mode=mode,
            action_mode=action_mode
        )

    def _gen_grid(self, width, height, start_pos=None, start_dir=None):
        
        width = 7 # hard-coded
        height = 11
        self.grid = Grid(height, height)
        wall = Wall()

        # Draw the left, center-column and right walls
        for i in range(0, height):
            self.grid.set(0, i, wall)
            self.grid.set(width-1, i, wall)
        # Draw the top, center-row and bottom walls
        for i in range(0, width):
            self.grid.set(i, 0, wall)
            self.grid.set(i, height-1, wall) 
        
        if self.randomize:
            self.start_pos = random.sample(self.valid_starts, 1)[0]
        else:
            self.start_pos = (3, 5)
        self.start_dir = 0

        self.mission = 'just roam around'

class VICGridEnv(VICEnv):
    def __init__(self):
        super().__init__(mode='grid')

class VICEnvOmni(VICEnv):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='omni')

class VICGridEnvOmni(VICEnv):
    def __init__(self):
        super().__init__(mode='grid', action_mode='omni')

class VICEnvRandomize(VICEnv):
    def __init__(self):
        super().__init__(randomize=True)

class VICGridEnvRandomize(VICEnv):
    def __init__(self):
        super().__init__(mode='grid', randomize=True)

class VICEnvOmniRandomize(VICEnv):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='omni', randomize=True)

class VICGridEnvOmniRandomize(VICEnv):
    def __init__(self):
        super().__init__(mode='grid', action_mode='omni', randomize=True)

register(
    id='MiniGrid-VICEnv-v0',
    entry_point='gym_minigrid.envs:VICEnv',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-v1',
    entry_point='gym_minigrid.envs:VICGridEnv',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-v2',
    entry_point='gym_minigrid.envs:VICEnvOmni',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-v3',
    entry_point='gym_minigrid.envs:VICGridEnvOmni',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-Randomize-v0',
    entry_point='gym_minigrid.envs:VICEnvRandomize',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-Randomize-v1',
    entry_point='gym_minigrid.envs:VICGridEnvRandomize',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-Randomize-v2',
    entry_point='gym_minigrid.envs:VICEnvOmniRandomize',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-VICEnv-Randomize-v3',
    entry_point='gym_minigrid.envs:VICGridEnvOmniRandomize',
    reward_threshold=1000.0
)
