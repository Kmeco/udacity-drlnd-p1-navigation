from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import pickle
import time


def dqn(n_episodes=1100, max_t=100000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                                 # list containing scores from each episode
    scores_window = deque(maxlen=100)           # last 100 scores
    eps = eps_start                             # initialize epsilon
    best_episode_so_far = 15

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]         # send the action to the environment
            next_state = env_info.vector_observations[0]    # get the next state
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]                   # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)             # save most recent score
        scores.append(score)                    # save most recent score
        eps = max(eps_end, eps_decay * eps)     # decrease epsilon
        avg_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))

        if avg_score >= best_episode_so_far:
            print('\nBest score reached in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         avg_score))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            best_episode_so_far = avg_score
    return scores


env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size=37, action_size=4, seed=0, hidden_layers=[128, 64, 32])
scores = dqn()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
plt.savefig('results/figure.png')
with open('results/scores-{}.pkl'.format(time.ctime()), 'wb') as f:
    pickle.dump(scores, f)
