# Reference code: https://github.com/gxnk/reinforcement-learning-code/
import gym
from policynet import PolicyGradient
import matplotlib.pyplot as plt
import time

DISPLAY_REWARD_THRESHOLD = 1000
RENDER = False

# Create environment
env = gym.make('Acrobot-v1')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,

)
# learning
for i_episode in range(85):
    observation = env.reset()
    while True:
        if RENDER: env.render()
        # sample actions and explore environment
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        # store observation, action and reward
        RL.store_transition(observation, action, reward)
        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99+ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
            print("episode:", i_episode, "rewards:", int(running_reward))
            # learn every episode
            vt = RL.learn()
            # plot state-action value
            if i_episode == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
# testing
for i in range(10):
    observation = env.reset()
    count = 0
    while True:
        env.render()
        # choose action greedily
        action = RL.greedy(observation)
        # action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if done:
            print(count)
            break
        observation = observation_
        count+=1
        print (count)

