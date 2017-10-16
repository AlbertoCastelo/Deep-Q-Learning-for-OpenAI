import gym
import math, random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

# maximum steps per episode
MAX_STEPS_EPI = 1000

class DQNController():

    def __init__(self, env, learning=False, testing=False, learning_rate=0.001, alpha=0.5, epsilon=0.5, 
        epsilon_thr=0.05, gamma=0.99, experience_size=3000, batch_size=16 ):

        self.done = False
        self.countSteps = 0
        self.nEpisodes = 0

        # valid actions
        self.valid_actions = [0, 1]

        '''
            Parameters of the agent
        '''
        self.env = env
        self.learning = learning
        self.testing = testing

        # learning rates
        self.learning_rate = learning_rate
        self.alpha = alpha

        # exploration-explotation
        self.epsilon = epsilon
        self.epsilon_thr = epsilon_thr

        # discount
        self.gamma = gamma

        # list to store the agent's experience
        self.experience = deque(maxlen=experience_size)
        self.batch_size = batch_size



        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # build neural network model
        self.model = self.define_nn()


    '''
    Defines Neural Network arquitecture for Deep Q Learning
    '''
    def define_nn(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def store(self, state, action, reward, state_previous, done):
        self.experience.append((state, action, reward, state_previous, done))

    '''
    - Resets the value of the environment to run a new episode
    - Updates the exploration-explotation rate
    '''
    def reset(self):
        self.env.reset()
        self.countSteps = 0
        self.nEpisodes += 1
        self.done = False

        # choose epsilon decay function
        decayF = 1

        if self.epsilon < self.epsilon_thr:
            # guarantee some exploration
            self.epsilon = self.epsilon_thr

        # decay epsilon each episode
        elif decayF == 0:
            # linear decay
            self.epsilon -= 0.05

        elif decayF == 1:
            # exponential decay
            aDecay = 0.8
            self.epsilon = aDecay ** self.nEpisodes

        elif decayF == 2:
            # quadratic inverse
            if self.nEpisodes == 0:
                self.epsilon = 1
            else:
                self.epsilon = self.nEpisodes ** (-2)

        elif decayF == 3:
            # exponential decay 2
            aDecay = 0.9
            self.epsilon = math.exp(-aDecay * self.nEpisodes)
            
        elif decayF == 4:
            aDecay = 0.1
            self.epsilon = math.cos(aDecay * self.nEpisodes)

        print("Epsilon: %f" % self.epsilon)


    '''
        Save and Load parameters for the neural network
    '''
    def load_model(self, filename):
        self.model.load_weights(filename)

    def save_model(self, filename):
        self.model.save_weights(filename)

    '''
    Chooses the action the agent will carry out
    '''
    def choose_action(self):

        if self.learning == False:
            # random action
            return random.choice(self.valid_actions)
        
        # sample exploration-explotation distribution
        sampleEE = random.random()
        if sampleEE <= self.epsilon:
            # random action
            return random.choice(self.valid_actions)

        # action based on highest Q-value
        action_val = self.model.predict(self.state)
        print(action_val)
        return np.argmax(action_val[0])

    '''
    Updates the neural network coefficients using minibatches
    '''
    def learn(self):
        if self.learning == True:
            if len(self.experience) <= self.batch_size:
                minibatch = random.sample(self.experience, len(self.experience))
            else:
                minibatch = random.sample(self.experience, self.batch_size)

            for state, action, reward, state_previous, done in minibatch:
                target = reward
                if not done:   #TRY WITHOUT THIS LINE
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(state)[0]))
                target_f = self.model.predict(state_previous)
                target_f[0][action] = target 
                self.model.fit(state_previous, target_f, epochs=1, verbose=0)

    def step(self):
        self.countSteps += 1
        # calculate action
        self.action = self.choose_action()

        # apply action and get feedback from environment
        self.state, self.reward, self.done, _ = self.env.step(self.action)

        # reshape state for input in NN
        self.state = np.reshape(self.state, [1, self.state_size])

        # highly penalize failing
        if not self.done:
            self.reward = -10.0

        # highly reward not having failed in 
        if self.countSteps > MAX_STEPS_EPI:
            self.reward = 10.0

        # store history for future replay
        if self.countSteps != 1:
            self.store(self.state, self.action, self.reward, self.state_previous, self.done)

        # learn
        self.learn()

        # update previous state
        self.state_previous = self.state 
    '''
    Runs an episode on the Environment
    '''
    def run(self):

        while (self.done == False or self.countSteps > MAX_STEPS_EPI):
            print("     Step: %d" % self.countSteps)
            if self.testing == True:
                self.env.render()
            
            # perform another simulation step
            self.step()