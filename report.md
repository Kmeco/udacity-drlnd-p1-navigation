[image1]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aece9b8_screen-shot-2018-05-04-at-6.14.42-pm/screen-shot-2018-05-04-at-6.14.42-pm.png 
[image2]: images/Screenshot2019.png
[image3]: images/myplot2.png
### Introduction
 This report sums up my solution of the unity banana collector environment. In the next sections, I explain the algorithm used, architecture of the Q-Network as well as the training process. 
 
### Learning Algorithm
 
##### Temporal-Difference Methods
Whereas Monte Carlo (MC) prediction methods must wait until the end of an episode to update the value function estimate, temporal-difference (TD) methods update the value function after every time step.

##### Q-Learning
Is an off-policy TD control method. It is guaranteed to converge to the optimal action value function q*, as long as the step-size parameter α is sufficiently small and ϵ is chosen to satisfy the Greedy in the Limit with Infinite Exploration (GLIE) conditions.
![algo][image1]

##### Q-Network architecture

* simple fully connected network with 3 hidden layers of `128, 64, 32` sizes.

- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network

##### Improvements

To address the inherent instabilities of Deep Q-Learning, I used two key features:
* Experience Replay
* Fixed Q-Targets

###### Experience replay 

> When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a replay buffer and using experience replay to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

> The replay buffer contains a collection of experience tuples (SS, AA, RR, S'S′). The tuples are gradually added to the buffer as we are interacting with the environment.

> The act of sampling a small batch of tuples from the replay buffer in order to learn is known as experience replay. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

###### Fixed Target network

> In Q-Learning, we update a guess with a guess, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters ww in the network q^ 
q^ to better approximate the action value corresponding to state SS and action AA with the following update rule:

![target][image2]
> where w- are the weights of a separate target network that are not changed during the learning step, and (SS, AA, RR, S'S′) is an experience tuple.

##### Results
Parameters are saved under `results/checkpoint.pt`

This plot displays episode # vs score over last 100 episodes

![rewards_plot][image3]

The number of episodes needed to reach an average reward (over 100 episodes) of at least +13: `540`

##### Ideas for Future Work

* Implement a double DQN, a dueling DQN, and prioritized experience replay