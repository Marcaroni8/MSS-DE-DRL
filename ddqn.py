import numpy as np
from collections import deque
# import itertools

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

class DDQNAgent:
    """Class for Double DQN agent."""

    def __init__(self, state_size, action_size):
        """Set parameters and build two identical models."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=int(1e6)) # 1e6 is typical (DRL 3.2.3.1 p.875)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.075
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self._copy_weights()

    def _build_model(self):
        """Neural Net for Deep-Q learning Model"""
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.state_size,)))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        self.loss = MeanSquaredError()
        model.compile(loss=self.loss,
                    optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _copy_weights(self):
        """copy online_model weigts to target_model"""
        self.target_model.set_weights(self.online_model.get_weights())

    def memorize(self, state, actions, reward, next_state, done):
        """Add a batch of observations to the memory."""
        for i in range(len(state)):
            self.memory.append([state[i], actions[i], reward[i], next_state[i], done]) # does done only apply to individual where optimum is reached?

    def act(self, state, Q=False):
        """Return a list of actions for a given list of states.
        online_model gives the Q-values for the actions for all states.
        Depending on epsilon, either the action with the maximum Q-values
        will be taken, or a random action.
        If Q=True, also return the Q-values.
        """
        NP = len(state)
        random_actions = np.random.randint(self.action_size, size=NP)
        if self.epsilon == 1.0:
            actions = random_actions
        else:
            p = self.online_model.predict_on_batch(state)
            actions = np.argmax(p, axis=1) # returns actions
            actions = np.where(np.random.random(NP) <= self.epsilon, random_actions, actions)
        if Q:
            return actions, p
        else:
            return actions

    def replay(self, batch_size):
        """Fetch a batch of random samples from self.memory, computer Q-values,
        compute target, train network and update epsilon.
        """
        # Fetch random samples from memory. Name this minibatch.
        sample = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in sample]

        # couldn't get slicing to work for list
        state = np.array([b[0] for b in minibatch])
        actions = np.array([b[1] for b in minibatch])
        reward = np.array([b[2] for b in minibatch])
        next_state = np.array([b[3] for b in minibatch])
        done = np.array([b[4] for b in minibatch])
        assert len(state) == batch_size
        assert len(actions) == batch_size
        assert len(reward) == batch_size
        assert len(next_state) == batch_size
        assert len(done) == batch_size

        # reshape states for model
        state = np.reshape(state, [batch_size, 1, self.state_size])
        next_state = np.reshape(next_state, [batch_size, 1, self.state_size])
        
        # compute Q-values using online model
        Q = self.online_model.predict_on_batch(next_state)
        # find actions for which Q is maximal for each sample
        a = np.argmax(Q, axis=1)
        # compute target using target model
        target_Q = self.target_model.predict_on_batch(next_state)[range(batch_size), a]
        discounted_reward = self.gamma * target_Q
        discounted_reward *= ~done
        target = reward + discounted_reward
        assert target.shape == (batch_size,)
        target_f = self.target_model.predict_on_batch(state)
        for i in range(batch_size):
            target_f[i,actions[i]] = target[i]
        assert len(target_f) == batch_size

        # train online model to minimize difference between predicted Q and target
        self.online_model.fit(state, target_f, epochs=1, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.loss(target_f, Q).numpy()

    def load(self, name):
        self.online_model.load_weights(name)
        self._copy_weights()

    def save(self, name):
        self.online_model.save_weights(name)