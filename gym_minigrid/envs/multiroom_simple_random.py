from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class MultiRoomSimpleEnvRandom(MiniGridEnv):
    """
    Environment with multiple rooms (no doors)
    """

    def __init__(self, mode='rgb', action_mode='forward'):

        super(MultiRoomSimpleEnvRandom, self).__init__(
            grid_size=11,
            max_steps=20000,
            see_through_walls=True,
            mode=mode,
            action_mode=action_mode
        )

    def _gen_grid(self, width, height, start_pos=None, start_dir=None):
        
        width = 11 # hard-coded
        height = 11
        self.grid = Grid(width, height)
        wall = Wall()
        
        vert_wall_loc = random.randint(width//4 + 1, (3*width)//4-1)
        horiz_wall_loc = random.randint(height//4 + 1, (3*height)//4-1)

        # Draw the left, center-column and right walls
        for i in range(0, height):
            self.grid.set(0, i, wall)
            if i != height//4 and i != (3*height)//4:
                self.grid.set(vert_wall_loc, i, wall)
            self.grid.set(width-1, i, wall)
        # Draw the top, center-row and bottom walls
        for i in range(0, width):
            self.grid.set(i, 0, wall)
            if i != height//4 and  i != (3*height)//4:
                self.grid.set(i, horiz_wall_loc, wall)
            self.grid.set(i, height-1, wall) 
        
        self.valid_starts = []
        for x in range(0, width-1):
            for y in  range(1, height-1):
                if self.grid.get(x, y) is None:
                    self.valid_starts.append((x, y))

        self.start_pos = random.sample(self.valid_starts, 1)[0]
        self.start_dir = 0

        self.mission = 'just roam around'

class MultiRoomSimpleEnvOmniRandom(MultiRoomSimpleEnvRandom):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='omni')

class MultiRoomSimpleGridEnvRandom(MultiRoomSimpleEnvRandom):
    def __init__(self):
        super().__init__(mode='grid', action_mode='forward')

class MultiRoomSimpleGridEnvOmniRandom(MultiRoomSimpleEnvRandom):
    def __init__(self):
        super().__init__(mode='grid', action_mode='omni')

register(
    id='MiniGrid-MultiRoom-SimpleRandom-v0',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnvRandom',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-SimpleRandom-v1',
    entry_point='gym_minigrid.envs:MultiRoomSimpleGridEnvRandom',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-SimpleRandom-v2',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnvOmniRandom',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-SimpleRandom-v3',
    entry_point='gym_minigrid.envs:MultiRoomSimpleGridEnvOmniRandom',
    reward_threshold=1000.0
)

