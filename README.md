# Online Mutation Strategy Selection in Differential Evolution through Deep Reinforcement Learning
This GitHub repository contains the code to recreate the data and plots from the thesis.

An artificial neural network is trained to dynamically select the mutation strategy in differential evolution for every individual in every generation.

## Files
 - `ddqn.py`: Contains the `DDQNAgent` class. This builds the neural network and provides all functions to train and use the model. The `load` and `save` functions can be used to load and save checkpoints.
 - `DE.py`: Implementation of DE with support for adaptive mutation strategies. Helper functions are based on the code from [Sharma et al.](https://github.com/mudita11/DE-DDQN) The state and reward function are implemented within the `DE` class. `reset` should always be run at the start of an episode. `step` moves the DE forward one generation, so it expects a list of `NP` actions, not one. Checkpoints and logs are made regularly.
 - `train.py`: Creates `DDQNAgent` and `DE` objects and trains the neural network using reinforcement learning.
 - `test.ipynb`: Jupyter Notebooks for recreating test data and (almost) all plots seen in the thesis. Note that this notebook is less organised.
 - `DE_oldFeatures.py`: Older version of `DE.py` which contains some more features that were later found to be identical to others. These have been removed in `DE.py`. This should not alter performance, but to open the checkpoints for `model1` and `model2`, use this file instead of `DE.py`.
 - `./modelX`: These three directories contain checkpoints of the three models referred to with the same name in the thesis.
 - `./figures`: All tables and plots that were created from the data.

## Usage
To start training a new model, simply run `train.py`. Alter variables within the files at will.