from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnvNoGoal(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8, mode='rgb'):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            mode=mode
        )
        self.start_pos_all = self._all_starts()
        self.dir_all = (0, 1, 2, 3)

    def _gen_grid(self, width, height, start_pos=None, start_dir=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        if start_pos is None:
            self.start_pos = (height//2, width//2)
        else:
            if start_pos in self.start_pos_all:
                self.start_pos = start_pos
            else:
                raise ValueError('(%d, %d) - Not a valid start position!'%(start_pos[0], start_pos[1]))

        if start_dir is None:
            self.start_dir = 0
        else:
            if start_dir in self.dir_all:
                self.start_dir = start_dir
            else:
                raise ValueError('%d - Not a valid direction!'%(start_dir))

        self.mission = "just roam around"
    
    def _all_starts(self):
        start_pos_all = []
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                start_pos_all.append((i, j))
        return start_pos_all

class EmptyEnvNoGoal6x6(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=6)

class EmptyEnvNoGoal16x16(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=16)

class EmptyEnvNoGoalSimple6x6(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=6, mode='grid')

class EmptyEnvNoGoalSimple8x8(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=8, mode='grid')

class EmptyEnvNoGoalSimple16x16(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=16, mode='grid')

register(
    id='MiniGrid-Empty-NoGoal-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnvNoGoal6x6'
)

register(
    id='MiniGrid-Empty-NoGoal-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnvNoGoal'
)

register(
    id='MiniGrid-Empty-NoGoal-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnvNoGoal16x16'
)

register(
    id='MiniGrid-Empty-NoGoal-Simple-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnvNoGoalSimple6x6'
)

register(
    id='MiniGrid-Empty-NoGoal-Simple-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnvNoGoalSimple8x8'
)

register(
    id='MiniGrid-Empty-NoGoal-Simple-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnvNoGoalSimple16x16'
)

