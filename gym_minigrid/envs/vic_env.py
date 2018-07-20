from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class VICEnv(MiniGridEnv):
    """
    Environment with 10x6 dimensions (9x5 effective states) 
    """

    def __init__(self, mode='rgb', action_mode='forward'):

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
