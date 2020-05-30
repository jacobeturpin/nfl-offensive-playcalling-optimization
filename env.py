"""NFL Playcalling Environment"""

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
		self.observation_space = spaces.Discrete(10)

		# Start the first drive
		self.reset()
    
	def step(self, action):
		assert self.action_space.contains(action)

		curr_obs = self._get_obs()
		new_pos = curr_obs + self.field_position
		# print(f"Step obs {curr_obs} added to {self.field_position} on down {self.remaining_downs}")
		# start by handling turnovers
		if self.remaining_downs <= 0 or curr_obs == 0:
			done = True
			reward = -1.
		# check if new field position is a touchdown
		elif new_pos >= 10:
			done = True
			reward = 1.
		# if not TO or TD then not done and no rewards
		else:
			done = False
			reward = 0.
		
		# decrement downs and change field position
		self.remaining_downs -= 1
		self.field_position = new_pos
		return new_pos, reward, done, {}

	def _get_obs(self):
		# random sampling from actions to be replaced by probablistic data
		obs = self.observation_space.sample()
		return obs

	def reset(self):
		self.field_position = 2
		self.remaining_downs = 3
		return self._get_obs()

	def render(self, mode='human'):

		print(f'Current Field Position: {self.field_position}')
		print(f'Remaining Downs: {self.remaining_downs}')