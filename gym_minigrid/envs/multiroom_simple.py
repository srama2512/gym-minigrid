from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class MultiRoomSimpleEnv(MiniGridEnv):
    """
    Environment with multiple rooms (no doors)
    """

    def __init__(self, randomize=False, mode='rgb'):

        super(MultiRoomSimpleEnv, self).__init__(
            grid_size=27,
            max_steps=20000,
            see_through_walls=True,
            mode=mode
        )
        # TODO
        self.randomize = randomize

    def _gen_grid(self, width, height, start_pos=None, start_dir=None):
        
        width = 27 # hard-coded
        height = 27
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

        self.start_pos = (12, 12)
        self.start_dir = 0

        self.mission = 'just roam around'

class MultiRoomSimpleEnvGrid(MultiRoomSimpleEnv):
    def __init__(self):
        super().__init__(mode='grid')

register(
    id='MiniGrid-MultiRoom-Simple-v0',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnv',
    reward_threshold=1000.0
)
register(
    id='MiniGrid-MultiRoom-Simple-v1',
    entry_point='gym_minigrid.envs:MultiRoomSimpleEnvGrid',
    reward_threshold=1000.0
)
