#!/usr/bin/env python3

from __future__ import division, print_function

import pdb
import sys
import numpy
import gym
import time
from optparse import OptionParser

import matplotlib.pyplot as plt
import gym_minigrid

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-NoGoal-Simple-8x8-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render(mode='human')
    
    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'CTRL':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)
        print('step=%s, reward=%.2f' % (env.step_count, reward))
        print([obs['image'][:, :, i] for i in range(3)])
        print(obs['image'].shape)
        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human', show_seen=False)
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
