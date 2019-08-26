
from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
from dqn_agent import Agent
import sys

def play_banana(isDoubleDQN=0):
    isDoubleDQN = int(isDoubleDQN)
    # find the path to the environment, this can be different for different OS
    env = UnityEnvironment(file_name="Banana_Windows_x86_64\Banana.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)

    # instantiate agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0, isDoubleDQN=isDoubleDQN)

    # load the weights from file
    if (isDoubleDQN==1):
        print("Using Double DQN")
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint_double_agent.pth'))
    else:
        print("Not Using Double DQN")
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint_simple_agent.pth'))  

    # start the agent
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, eps=0)      # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))

    env.close()



if __name__ == "__main__":
    play_banana(sys.argv[1])


