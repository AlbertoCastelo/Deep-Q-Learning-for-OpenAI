import gym
import math, random
from DQNController import DQNController



def main():
    env = gym.make('CartPole-v0')
    agent = DQNController(env=env, learning=True, testing=False, learning_rate=0.001, alpha=0.5, epsilon=0.5, 
        epsilon_thr=0.05, gamma=0.99, experience_size=30000, batch_size=64)

    i = 0
    # for each training episodes
    while agent.epsilon > agent.epsilon_thr:
        i += 1
        print("Episode %d" % i)
        # reset environment conditions
        agent.reset()

        # run another episode
        agent.run()

    
    agent.save_model('models/nn_coefficients.h5')


main()
