"""NFL Playcalling Environment"""
import random

import gym
from gym import spaces


import data_loader as nfl_data

# Create some test functions until api is built


class NFLPlaycallingEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(NFLPlaycallingEnv, self).__init__()

		# Get data for probabilistic data
		filename = './data/nfl-play-by-play.csv'
		with open('data/dtypes.txt', 'r') as inf:
			dtypes_dict = eval(inf.read())

		self.FieldPos = nfl_data.FieldPositionLoader(filename, dtypes_dict=dtypes_dict)

		# self._get_field_pos()
		# Define action and observation space
		# They must be gym.spaces objects

		# three discrete actions - pass, run, qb sneak
		self.action_space = spaces.Discrete(3)

		# observation space: field position, down, to_go, turnover, touchdown
		self.observation_space = spaces.Tuple((
			spaces.Discrete(100), #field position
			spaces.Discrete(4), #down
			spaces.Discrete(99), #to_go
			spaces.Discrete(2), #turnover
			spaces.Discrete(2))) #touchdown


	def step(self, action):
		assert self.action_space.contains(action)

		obs = self._get_observation(action)

		# print(f"Step obs {obs} added to {self.field_position} remaining downs {self.remaining_downs}")

		# start by handling turnovers
		if obs[1] <= 0 or obs[3] == 1:
			print(f"turnover {obs}")
			done = True
			reward = -1.
		# check if new field position is a touchdown
		elif obs[4] == 1:
			print(f"td {obs}")
			done = True
			reward = 1.
		# if not TO or TD then not done and no rewards
		else:
			print(f"continue {obs}")
			done = False
			reward = 0.
		
		return obs, reward, done, {}

	def _get_observation(self, action):
		# TODO: replace with probabilistic outcomes
		outcomes = self._get_field_pos(action)
		outcome_idx = random.choices([i for i, x in enumerate(outcomes)], weights=[x[2] for x in outcomes])
		outcome = outcomes[outcome_idx[0]]
		# print(f"outcome {outcome}")

		if outcome[0] == 'BALL_MOVED':
			# ball moved
			if (outcome[1] + self.field_position) >= 100:
				self.field_position = 100
				self.touchdown = 1
			elif outcome[1] >= self.to_go:
				self.remaining_downs = 4 # well get decremented to 3 below
				self.to_go = 10 #TODO: add goalline situations
			else:
				self.to_go -= outcome[1]
			self.field_position = self.field_position + outcome[1]
		elif outcome[0] == "INTERCEPTION" or outcome[0] == "FUMBLE":
			# turnover
			self.turnover = 1
		elif outcome[0] == "TOUCHDOWN":
			# touchdown
			self.field_position = 100
			self.touchdown = 1
		else:
			print("INVALID ACTION")

		# decrement downs
		self.remaining_downs -= 1

		# print(f"updates: yardline:{self.field_position} turnover:{self.turnover} td:{self.touchdown}")
		return (self.field_position, self.remaining_downs, self.to_go, self.turnover, self.touchdown)

	def _gen_rand_outcomes(self):
		
		outcomes = []
		for i in range(4):
			outcomes.append(('BALL_MOVED', random.randint(0,self.to_go*2), 0.2))
		outcomes.append(('INTERCEPTION', -5, 0.1))
		outcomes.append(('TOUCHDOWN', 100-self.field_position, 0.1))

		return outcomes
	def _get_field_pos(self, action):
		if action == 0:
			action_val = nfl_data.PlayType.PASS
		elif action == 1:
			action_val = nfl_data.PlayType.RUN
		elif action == 2:
			action_val = nfl_data.PlayType.QB_SNEAK
		else:
			raise ValueError('invalid action')
		# print(f"Action Taken {action_val}")

		outcomes = self.FieldPos.get_probability(down = self.remaining_downs, 
			to_go = self.to_go, 
			position = 100-self.field_position, 
			play = action_val
			)

		return outcomes


	def reset(self):

		self.field_position = 20
		self.remaining_downs = 3
		self.to_go = 10
		self.turnover = 0
		self.touchdown = 0

		return (self.field_position, self.remaining_downs, self.to_go, self.turnover, self.touchdown)

	def render(self, mode='human'):

		print(f'Current Field Position: {self.field_position}')
		print(f'Remaining Downs: {self.remaining_downs}')