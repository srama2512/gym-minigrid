from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class MultiRoomSimpleEnv(MiniGridEnv):
    """
    Environment with multiple rooms (no doors)
    """

    def __init__(self, randomize=False, mode='rgb', action_mode='forward'):

        self.randomize = randomize
        super(MultiRoomSimpleEnv, self).__init__(
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

        # Draw the left, center-column and right walls
        for i in range(0, height):
            self.grid.set(0, i, wall)
            if i != height//4 and i != (3*height)//4:
                self.grid.set(width//2, i, wall)
            self.grid.set(width-1, i, wall)
        # Draw the top, center-row and bottom walls
        for i in range(0, width):
            self.grid.set(i, 0, wall)
            if i != height//4 and  i != (3*height)//4:
                self.grid.set(i, height//2, wall)
            self.grid.set(i, height-1, wall) 
        
        self.valid_starts = []
        for x in range(0, width-1):
            for y in  range(1, height-1):
                if self.grid.get(x, y) is None:
                    self.valid_starts.append((x, y))

        if self.randomize:
            self.start_pos = random.sample(self.valid_starts, 1)[0]
        else:    
            self.start_pos = (4, 4)
        self.start_dir = 0

        self.mission = 'just roam around'

class MultiRoomSimpleEnvOmni(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='omni', randomize=False)

class MultiRoomSimpleGridEnv(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='grid', action_mode='forward', randomize=False)

class MultiRoomSimpleGridEnvOmni(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='grid', action_mode='omni', randomize=False)

class MultiRoomSimpleEnvRandomize(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='forward', randomize=True)

class MultiRoomSimpleEnvOmniRandomize(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='rgb', action_mode='omni', randomize=True)

class MultiRoomSimpleGridEnvRandomize(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='grid', action_mode='forward', randomize=True)

class MultiRoomSimpleGridEnvOmniRandomize(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='grid', action_mode='omni', randomize=True)

register(
    id='MiniGrid-MultiRoom-Simple-v0',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnv',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-v1',
    entry_point='gym_minigrid.envs:MultiRoomSimpleGridEnv',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-v2',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnvOmni',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-v3',
    entry_point='gym_minigrid.envs:MultiRoomSimpleGridEnvOmni',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-Randomize-v0',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnvRandomize',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-Randomize-v1',
    entry_point='gym_minigrid.envs:MultiRoomSimpleGridEnvRandomize',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-Randomize-v2',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnvOmniRandomize',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-Randomize-v3',
    entry_point='gym_minigrid.envs:MultiRoomSimpleGridEnvOmniRandomize',
    reward_threshold=1000.0
)
