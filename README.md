# DRL Banana Navigation
 Deep Reinforcement Learning AI - collecting yellow bananas

 This project is part of Udacity's Nanodegree on Deep Reinforcement Learning (https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## The Banana collector AI

The goal of this project is to train an agent (AI) to navigate a virtual world where it can collect blue and yellow bananas. The task of the agent is to collect as many yellow bananas as possible, while avoiding the blue bananas.

In the animatation bellow you can see my trained agent navigating in the world.

![Banana World](https://github.com/sinusgamma/DRL-Banana-Navigation/blob/master/banana_navigation.gif)

I trained two agents. One is a Deep Q-Network DQN with "experience replay" and "fixed § targets". The other algorithm uses Double DQN as well. In my case there wasn't significant difference between the performance of these two solutions. 

The exercise uses the Unity Machine Learning Agents Toolkit (https://github.com/Unity-Technologies/ml-agents)
Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.<br>
1 - move backward.<br>
2 - turn left.<br>
3 - turn right.<br>

The task is episodic, and in order to solve the environment, the agent had to get an average score of +13 over 100 consecutive episodes.

## The Environment

### Step 1: Clone the DRLND Repository
To set up the environment, please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in [README.md]((https://github.com/udacity/deep-reinforcement-learning#dependencies)) at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Step 2: Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)<br>
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)<br>
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)<br>
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)<br>

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

## Testing the trained AI

After setting up the environment you can test the DQN and Double DQN algorithm and watch the agents collecting bananas. Calling ```load_watch_ai.py 0``` will call the non-DoubleDQN agent, while ```load_watch_ai.py 1``` will call the DoubleDQN agent.