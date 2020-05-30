"""NFL Playcalling Environment"""
import random

import gym
from gym import spaces



# import data_loader as nfl_data

# Create some test functions until api is built


class NFLPlaycallingEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(NFLPlaycallingEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects

		# three discrete actions - pass, run, qb sneak
		self.action_space = spaces.Discrete(3)

		# ten discrete observations - 10...80, TD, TO
		self.observation_space = spaces.Tuple((
			spaces.Discrete(100), 
			spaces.Discrete(2), 
			spaces.Discrete(2)))

    
	def step(self, action):
		assert self.action_space.contains(action)

		obs = self._get_outcome(action)

		print(f"Step obs {obs} added to {self.field_position} remaining downs {self.remaining_downs}")

		# start by handling turnovers
		if self.remaining_downs <= 0 or obs[1] == 1:
			print(f"turnover")
			done = True
			reward = -1.
		# check if new field position is a touchdown
		elif obs[2] == 1:
			print(f"td")
			done = True
			reward = 1.
		# if not TO or TD then not done and no rewards
		else:
			print(f"continue")
			done = False
			reward = 0.
		
		return obs, reward, done, {}

	def _get_outcome(self, action):
		# TODO: replace with probabilistic outcomes
		outcome = random.choices([1,2,3,4], weights=[0.8, 0.05, 0.05, 0.1])
		outcome = outcome[0]

		if outcome == 1:
			# ball moved
			self.field_position = self.field_position + random.randint(0,99-self.field_position) #TODO: replace with field placement from probablistic data
			print(f"new position {self.field_position}")
		elif outcome == 2 or outcome == 3:
			# turnover
			self.turnover = 1
		elif outcome == 4:
			# touchdown
			self.field_position = 100
			self.touchdown = 1
		else:
			print("invalid action")

		# decrement downs
		self.remaining_downs -= 1

		print(f"updates: yardline:{self.field_position} turnover:{self.turnover} td:{self.touchdown}")
		return (self.field_position, self.turnover, self.touchdown)

	# def _next_obs(self):
	# 	curr_obs = self._get_obs()
	# 	new_pos = curr_obs + self.field_position
		

	# def _get_obs(self):
	# 	# random sampling from actions to be replaced by probablistic data
	# 	obs = self.observation_space.sample()
	# 	return obs

	def reset(self):
		self.remaining_downs = 3

		self.field_position = 20
		self.turnover = 0
		self.touchdown = 0

		return (self.field_position, self.turnover, self.touchdown)

	def render(self, mode='human'):

		print(f'Current Field Position: {self.field_position}')
		print(f'Remaining Downs: {self.remaining_downs}')