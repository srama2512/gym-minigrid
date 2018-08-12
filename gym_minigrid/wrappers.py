import pdb
import math
import operator
from functools import reduce
import cv2
import numpy as np

import gym
from gym import error, spaces, utils

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (env.agentPos, env.agentDir, action)

        # Get the count for this (s,a) pair
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this (s,a) pair
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (env.agentPos)

        # Get the count for this key
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this key
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class ScaledObsWrapper(gym.core.ObservationWrapper):
    """
    Take input observation image (256x256x3) and scale it down to
    84x84x1. Ignores mission strings.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype='uint8'
        )
        self.actions = env.actions

    def observation(self, obs):
        # obs = 256x256x3 array
        image = obs
        image_gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (84, 84))
        return image_gray[:, :, np.newaxis]

    @property
    def step_count(self):
        return self.env.step_count

class SimpleFlatObsWrapper(gym.core.ObservationWrapper):
    """
    Converts 84x84x1 image into a 7056 vector
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(7056,),
            dtype='uint8'
        )
        self.actions = env.actions

    def observation(self, obs):
        return obs.flatten()

    @property
    def step_count(self):
        return self.env.step_count

class PosDirFlatWrapper(gym.core.ObservationWrapper):
    """
    Returns just the position, direction as flattened array
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3,),
            dtype='float32'
        )
        self.actions = env.actions

    def observation(self, obs):
        return np.array([obs['position'][0], obs['position'][1], obs['direction']])

    @property
    def step_count(self):
        return self.env.step_count

class PosDirObsFlatWrapper(gym.core.ObservationWrapper):
    """
    Returns the position, direction and observation grid as flattened array
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3 + reduce((lambda x, y: x* y), self.observation_space.spaces['image'].shape), ),
            dtype='float32'
        )
        self.actions = env.actions

    def observation(self, obs):
        # width x height x 3 array
        pos_dir = np.array([obs['position'][0], obs['position'][1], obs['direction']])
        pos_dir_obs = np.concatenate([obs['image'].flatten(), pos_dir], axis=0)

        return pos_dir_obs
   
    @property
    def step_count(self):
        return self.env.step_count

class ScaledObsWrapper_v2(gym.core.ObservationWrapper):
    """
    Take input observation image and scale it down to
    128x128x3. Ignores mission strings.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(128, 128, 3),
            dtype='uint8'
        )
        self.actions = env.actions

    def observation(self, obs):
        # obs = 256x256x3 array
        image = obs
        image = cv2.resize(image, (128, 128))
        return image

    @property
    def step_count(self):
        return self.env.step_count



class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=64):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )
        self.actions = env.actions

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, "mission string too long"
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

    @property
    def step_count(self):
        return self.env.step_count

