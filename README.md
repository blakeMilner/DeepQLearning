DeepQLearning
=============

Written by Blake Milner and Jeff Soldate, with help from Eugenio Culurciello and his lab. Work was
done as part of a project for BME495, a Computational Neuroscience course at Purdue. The original
code, written in JavaScript, was developed by Andrej Karpathy, a Ph.D. student at Stanford University.

Deep Q Learning is a powerful machine learning algorithm utilizing Q-Learning. The state space is implemented using Neural Networks, thus bypassing inefficient static look up tables. This aplication was implemented using Torch 7 and Lua.

In many practical engineering scenarios it is often necessary for an algorithm to perform a
series of decisions in order to accomplish a given task. However, that task itself is not always
well-defined and the intermediate decisions to accomplish it are often complex and ever-changing.
Furthermore, information that contributes to accomplishing the task is often not readily available
until critical intermediate decisions have already been made. Video games are a good example of
situations in which a series of actions is required in order to accomplish a task.

This application also presents an AI based approach to learning a game where the rules aren't
immediately known. In recent years very robust algorithms utilizing these concepts have been developed and applied 
successfully to retro Atari video games: http://arxiv.org/pdf/1312.5602v1.pdf.

Reinforcement learning methods that encourage both exploration and strategizing have been developed in
order to address this problem. One of these methods, called Q-learning, utilizes a policy in order to
select an optimal action.

The Q-learning algorithm hinges on a utility function called the Q-function. This function
accepts a state that contains all pertinent information about the playing field along with a possible
action. The function returns a number that describes the utility of that action. In Q-learning the utility
of an action is evaluated based on the immediate reward gained from taking that action and the
possibility of a delayed reward that the action may lead to. For large games with many states and possible
actions the above approach is very time-consuming and computationally intense. Using a neural network to
represent the Q-function can solve many of these issues by eliminating the need for enumeration in order to completely
support the exploration of the state space.

An implementation of the method described above (written in JavaScript) exists and is freely available:
http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html

However, this package is designed for a browser and used primarily as a learning tool. DeepQLearning is a 
partial port of the Q-learning component of this package to the Lua scripting language. The Neural Network 
component is powered by Torch 7, a scientific computing framework used for machine learning. It is the hope 
of the authors that this package can be used to fuel further scientific inquiry into this topic.

This page also contains a broswer game that the JS Qlearning engine learns from scratch. If the settings are optimized
then after about 15 minutes the application will have learned to play the game flawlessy.


Installation and Use
====================

Requirements:

 * Torch7 (with nnx and optim package) 
-- A scientific computing framework with wide support for machine learning algorithms. (https://github.com/torch/torch7)


Usage:

The DeepQLearning module can be easily included in a Lua scipt using:

```bash
Brain = require 'deepqlearn'
```

The brain must then be initialized with the number of expected inputs and outputs:

```bash
Brain.init(num_inputs, num_outputs)   
```

An action can be selected from an input state space using:

```bash
action = Brain.forward(state); 
```

Learning can be affected from the last state space input to Brian.forward by giving a reward value:

```bash
Brain.backward(reward); 
```
