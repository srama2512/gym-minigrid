from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnvNoGoal(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8, mode='rgb', action_mode='forward'):
        super().__init__(
            grid_size=size,
            max_steps=10000,#4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            mode=mode,
            action_mode=action_mode
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
        self.valid_starts = self._all_starts()
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

# Normal forward mode
class EmptyEnvNoGoal6x6(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=6)

class EmptyEnvNoGoal16x16(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=16)

# Grid versions of forward mode
class EmptyGridEnvNoGoal6x6(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=6, mode='grid')

class EmptyGridEnvNoGoal8x8(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=8, mode='grid')

class EmptyGridEnvNoGoal16x16(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=16, mode='grid')

# Normal omni mode
class EmptyEnvOmniNoGoal6x6(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=6, action_mode='omni')

class EmptyEnvOmniNoGoal8x8(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=8, action_mode='omni')

class EmptyEnvOmniNoGoal16x16(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=16, action_mode='omni')

# Grid versions of omni mode
class EmptyGridEnvOmniNoGoal6x6(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=6, mode='grid', action_mode='omni')

class EmptyGridEnvOmniNoGoal8x8(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=8, mode='grid', action_mode='omni')

class EmptyGridEnvOmniNoGoal16x16(EmptyEnvNoGoal):
    def __init__(self):
        super().__init__(size=16, mode='grid', action_mode='omni')

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
    id='MiniGrid-Empty-NoGoal-6x6-v1',
    entry_point='gym_minigrid.envs:EmptyGridEnvNoGoal6x6'
)

register(
    id='MiniGrid-Empty-NoGoal-8x8-v1',
    entry_point='gym_minigrid.envs:EmptyGridEnvNoGoal'
)

register(
    id='MiniGrid-Empty-NoGoal-16x16-v1',
    entry_point='gym_minigrid.envs:EmptyGridEnvNoGoal16x16'
)

register(
    id='MiniGrid-Empty-NoGoal-6x6-v2',
    entry_point='gym_minigrid.envs:EmptyEnvOmniNoGoal6x6'
)

register(
    id='MiniGrid-Empty-NoGoal-8x8-v2',
    entry_point='gym_minigrid.envs:EmptyEnvOmniNoGoal8x8'
)

register(
    id='MiniGrid-Empty-NoGoal-16x16-v2',
    entry_point='gym_minigrid.envs:EmptyEnvOmniNoGoal16x16'
)

register(
    id='MiniGrid-Empty-NoGoal-6x6-v3',
    entry_point='gym_minigrid.envs:EmptyGridEnvOmniNoGoal6x6'
)

register(
    id='MiniGrid-Empty-NoGoal-8x8-v3',
    entry_point='gym_minigrid.envs:EmptyGridEnvOmniNoGoal'
)

register(
    id='MiniGrid-Empty-NoGoal-16x16-v3',
    entry_point='gym_minigrid.envs:EmptyGridEnvOmniNoGoal16x16'
)


