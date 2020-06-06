
from env import NFLPlaycallingEnv

def random_play(environment, episodes=5, render=False):

	# for ep in range(episodes):

	total_reward = 0.0
	total_steps = 0
	obs = environment.reset()

	while True:
		action = environment.action_space.sample()
		obs, reward, done, _ = environment.step(action)

		if render:
			env.render()

		total_reward += reward
		total_steps += 1

		# print(obs, reward, done)
		if done:
			break

	print("Episode done in {} steps with {:.2f} reward".format(total_steps, total_reward))

def test_play(environment):
	obs = environment.reset()
	print(obs)
	# obs, reward, done, _ = environment.step(4) # something isn't right with punting from the 20
	# environment._set_field_pos(field_position=80, remaining_downs=1)
	obs, reward, done, _ = environment.step(3)

if __name__ == '__main__':
	env = NFLPlaycallingEnv()

	print(env.observation_space.shape)
	print (env.observation_space)
	random_play(env, render=True)
	# test_play(env)