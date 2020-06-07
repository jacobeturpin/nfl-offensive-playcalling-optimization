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
		self.action_space = spaces.Discrete(5)

		# observation space: field position, down, to_go, turnover, touchdown
		self.observation_space = spaces.Tuple((
			spaces.Discrete(100), #field position
			spaces.Discrete(4), #down
			spaces.Discrete(99), #to_go
			spaces.Discrete(2), #turnover
			spaces.Discrete(2),#touchdown
			spaces.Discrete(2))) # field_goal

		self.action_dict = {
			0: 'PASS',
			1: 'RUN',
			2: 'QB_SNEAK',
			3: 'FIELD_GOAL',
			4: 'PUNT'
		}


	def step(self, action):
		"""Increment the environment one step given an action

		Attributes:
			action (int): 0-6 value specifying the action taken

		Returns:
			obs, reward, done, {} (Tuple): observations, current reward for step, and done flag
		"""
		assert self.action_space.contains(action)

		obs = self._get_observation(action)
		
		# check if observation state is a touchdown
		if obs[4] == 1:
			# print(f"action {self.action_dict[action]} td {obs}")
			done = True
			reward = 7.
		# check if observation state is a field goal
		elif obs[5] == 1:
			# print(f"action {self.action_dict[action]} field goal {obs}")
			done = True
			reward = 3.
		# check if it is a turnover
		elif obs[1] <= 0 or obs[3] == 1:
			# print(f"action {self.action_dict[action]} turnover {obs}")
			done = True
			reward = -7. * (1 - obs[0]/100)
		# if not TO or TD then not done and no rewards
		else:
			# print(f"action {self.action_dict[action]} continue {obs}")
			done = False
			reward = 0.
		
		print(f'state: action {self.action_dict[action]}, obs: {obs}, done: {done}, reward: {reward}')
		return obs, reward, done, {}

	def _get_observation(self, action):
		"""Calculate the observation space using historical outcomes based on the action taken

		Attributes:
			action (int): 0-6 value specifying the action taken

		Returns:
			obs (Tuple of Discreet): the current observation space after the action has been applied
		"""
		
		# get outcomes from historical data
		outcomes = self._get_field_pos(action)
		try: 
			outcome_idx = random.choices([i for i, x in enumerate(outcomes)], weights=[x[2] for x in outcomes])
			outcome = outcomes[outcome_idx[0]]
		except:
			print(f"NO OUTCOMES: action {action}, outcomes: {outcomes}")
			outcome = nfl_data.PlayOutcome(type='BALL_MOVED', yards=0.0, prob=1)

		if outcome[0] == 'BALL_MOVED':
			# update field position for any BALL_MOVED outcome
			self.field_position = self.field_position + outcome[1]

			# ball moved
			if action == 4:
				#punted
				self.turnover = 1
			elif self.field_position >= 100:
				# implied touchdown
				self.field_position = 100
				self.touchdown = 1
			elif outcome[1] >= self.to_go:
				# first down
				self.remaining_downs = 4 # will get decremented to 3 below
				self.to_go = 100 - self.field_position if self.field_position >= 90 else 10
			else:
				# move the ball and decrement the down
				self.to_go -= outcome[1]

		elif outcome[0] == 'INTERCEPTION' or outcome[0] == 'FUMBLE':
			# turnover
			self.turnover = 1
			self.field_position = self.field_position + outcome[1]
		elif outcome[0] == 'TOUCHDOWN':
			# touchdown
			self.field_position = 100
			self.touchdown = 1
		elif outcome[0] == 'FIELD_GOAL_MADE':
			# field goal was made
			self.field_goal = 1
		elif outcome[0] == 'FIELD_GOAL_MISSED':
			# field goal was missed
			self.turnover = 1
			self.field_position = self.field_position + outcome[1]
		else:
			raise ValueError('invalid action')

		# decrement downs
		self.remaining_downs -= 1

		# print(f"updates: yardline:{self.field_position} turnover:{self.turnover} td:{self.touchdown}")
		return self._return_obs_state()
	def _return_obs_state(self):
		"""Return the observation space at a given time
		"""
		return (self.field_position, self.remaining_downs, self.to_go, self.turnover, self.touchdown, self.field_goal)

	def _gen_rand_outcomes(self):
		
		outcomes = []
		for i in range(4):
			outcomes.append(('BALL_MOVED', random.randint(0,self.to_go*2), 0.2))
		outcomes.append(('INTERCEPTION', -5, 0.1))
		outcomes.append(('TOUCHDOWN', 100-self.field_position, 0.1))

		return outcomes
	def _get_field_pos(self, action):
		"""Given an action, return the outcome based on the likelihood from historical data

		Attributes:
				action (int): Number associated with action taken for discrete observation space. See if statement for number coding
		"""
		if action == 0:
			action_val = nfl_data.PlayType.PASS
		elif action == 1:
			action_val = nfl_data.PlayType.RUN
		elif action == 2:
			action_val = nfl_data.PlayType.QB_SNEAK
		elif action == 3:
			action_val = nfl_data.PlayType.FIELD_GOAL
		elif action == 4:
			action_val = nfl_data.PlayType.PUNT
		else:
			raise ValueError('invalid action')
		# print(f"Action Taken {action_val}")

		outcomes = self.FieldPos.get_probability(down = self.remaining_downs, 
			to_go = self.to_go, 
			position = 100-self.field_position, 
			play = action_val
			)

		return outcomes

	def _set_field_pos(self, field_position = 20, remaining_downs = 3, to_go = 10, turnover = 0, touchdown = 0, field_goal = 0):
		"""Used for testing to set different scenarios

		Attributes:
				field_position (int): 0-100 value of field position where 20 is own 20 and 80 is opp 20
				remaining_downs (int): remaining downs before turnover
				to_go (int): distance to go for first down
				turnover (int): terminal state flag where 0=no turnover, 1=turnover
				touchdown (int): terminal state flag where 0=no touchdown, 1=touchdown
		"""
		self.field_position = field_position
		self.remaining_downs = remaining_downs
		self.to_go = to_go
		self.turnover = turnover
		self.touchdown = touchdown
		self.field_goal = field_goal

	def reset(self):

		self._set_field_pos()

		return self._return_obs_state()

	def render(self, mode='human'):

		print(f'Current Field Position: {self.field_position}')
		print(f'Remaining Downs: {self.remaining_downs}')