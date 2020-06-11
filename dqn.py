import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import pandas as pd

random.seed(0)

import gym
from env import NFLPlaycallingEnv
# env = gym.wrappers.Monitor(env, './runs', False, True)
import matplotlib.pyplot as plt
from collections import deque
import time

from tensorboardX import SummaryWriter

class DQNAgent():
    """
		Agent for DQN to be used for NFLPlaycallingEnv
		Referenced for DQN on Tuple Observation space: https://github.com/ml874/Blackjack--Reinforcement-Learning
		"""

    def __init__(self, env, epsilon=1.0, alpha=0.5, gamma=0.9, time = 30000):
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = env.observation_space
        self.memory = deque(maxlen=2000) # Record past experiences- [(state, action, reward, next_state, done)...]
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.gamma = gamma       # Discount factor- closer to 1 learns well into distant future
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning = True
        self.model = self._build_model()
        
        self.time = time 
        self.time_left = time # Epsilon Decay
        self.small_decrement = (0.4 * epsilon) / (0.3 * self.time_left) # reduce epsilon
        print('Model Initialized')
    
    # Build Neural Net
    def _build_model(self):
        """Create the model using Keras

        Returns:
          model (keras architecture): keras object specifying the model architecture
		    """
        model = Sequential()
        model.add(Dense(32, input_shape = (len(self.state_size)-3,), kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.alpha))
        
        return model
       

    def choose_action(self, state):
        """Choose an action based. Exploration if random is below epsilon, best predicted action exploitation if not (greedy)
        
        Attributes:
          state (np.Array): given current state as an array

        Returns:
          action (int): the action to be taken
		    """

        # if random number > epsilon, act 'rationally'. otherwise, choose random action
        
        if np.random.rand() <= self.epsilon:
            
            action = random.randrange(self.action_size)
            
        else:
            action_value = self.model.predict(state)

            action = np.argmax(action_value[0])
        
        self.update_parameters()
        return action

    def evaluation(self, state):
        """Choose an action based. Exploration if random is below epsilon, best predicted action exploitation if not (greedy)
        
        Attributes:
          state (np.Array): given current state as an array

        Returns:
          action (int): the action to be taken
		    """
        
        print('=====================================')
        print('Evaluation')
        print('=====================================')
        action_value = self.model.predict(state)
        action = np.argmax(action_value[0])

        print(f'For state: {state}, action is {action}')
			
			
			
			
			
        
    def update_parameters(self):
        """Update epsilon and alpha after each action. Set them to 0 if not learning
		    """

        if self.time_left > 0.9 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.7 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.5 * self.time:
            self.epsilon -= self.small_decrement

        elif self.time_left > 0.3 * self.time:

            self.epsilon -= self.small_decrement
        elif self.time_left > 0.1 * self.time:
            self.epsilon -= self.small_decrement

        self.time_left -= 1       


    def learn(self, state, action, reward, next_state, done):
        """Choose an action based. Exploration if random is below epsilon, best predicted action exploitation if not (greedy)
        
        Attributes:
          state (np.Array): given current state as an array
          action (int): action to be taken
          reward (float): current reward
          next_state (np.Array): given state as an array after step has been taken
          done (bool): flag if the episode is done
		    """
        
        target = reward

        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)


        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_weights(self, location):

      # Save the weights
      self.model.save_weights(location)
        



if __name__ == '__main__':
	start_time = time.time()
	env = NFLPlaycallingEnv()
	obs_size = len(env.observation_space)-3

	writer = SummaryWriter(comment="-dqn")

	num_rounds = 100 # Payout calculated over num_rounds
	num_samples = 50 # num_rounds simulated over num_samples

	agent = DQNAgent(env=env, epsilon=1.0, alpha=0.001, gamma=0.1, time=7500)

	average_payouts = []

	state = env.reset()
	state = np.reshape(state[0:obs_size], [1,obs_size])
	
	best_reward = -7 # store the best total reward across samples

	for sample in range(num_samples):
			round = 1
			total_payout = 0 # store total payout per sample
			while round <= num_rounds:
					action = agent.choose_action(state)
					next_state, payout, done, _ = env.step(action)
					next_state = np.reshape(next_state[0:obs_size], [1,obs_size])

					
					total_payout += payout    
	#         if agent.learning:
					agent.learn(state, action, payout, next_state, done)
					
					state = next_state
					state = np.reshape(state[0:obs_size], [1,obs_size])
					
					if done:
							state = env.reset() # Environment deals new cards to player and dealer
							state = np.reshape(state[0:obs_size], [1,obs_size])
							round += 1

			average_payouts.append(total_payout)

			reward = total_payout/num_rounds

			writer.add_scalar("reward", reward, sample)

      
			if reward > best_reward:
				print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
				print('=====================================')
				best_reward = reward

			writer.add_scalar("best_reward", best_reward, sample)

			if sample % 1 == 0:
					print('Done with sample: ' + str(sample) + str("   --- %s seconds ---" % (time.time() - start_time)))
					print(f"reward {reward}, best reward {best_reward}")
					print(agent.epsilon)
    
			
	print ("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts)/(num_samples)))
	
	agent.evaluation(np.array([50,1,15]).reshape(1,3))
	agent.evaluation(np.array([99,0,1]).reshape(1,3))
	agent.evaluation(np.array([30,0,10]).reshape(1,3))

		