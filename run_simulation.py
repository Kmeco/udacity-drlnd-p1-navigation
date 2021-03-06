import torch
from dqn_agent import Agent
from unityagents import UnityEnvironment


agent = Agent(state_size=37, action_size=4, seed=5, hidden_layers=[128, 64, 32])
agent.qnetwork_local.load_state_dict(torch.load('results/checkpoint.pth', map_location='cpu'))

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


for i in range(3):
    for j in range(3):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score

    while True:
        action = agent.act(state)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state

        reward = env_info.rewards[0]  # get the reward

        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        print('\rScore {}\t'.format(score), end="")
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))