import gymnasium as gym
import pandas as pd
from collections import defaultdict
import random
import tqdm

epsilon = 0.5

def epsilon_greedy_policy(state, Q):
    # set the epsilon value to 0.5

    # sample a random value from the uniform distribution, if the sampled value is less than
    # epsilon then we select a random action else we select the best action which has maximum Q
    # value as shown below

    # There are two actions: stick (0), and hit (1).
    # Stop draging cards if the player's sum is 20 or more.
    if state[0] >= 20:
        return 0

    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])

def generate_episode(num_timesteps, Q):
    episode = []
    state, _ = env.reset()

    for t in range(num_timesteps):

        # select the action according to the epsilon-greedy policy
        action = epsilon_greedy_policy(state, Q)

        # perform the selected action and store the next state information
        next_state, reward, done, info, _ = env.step(action)

        # store the state, action, reward in the episode list
        episode.append((state, action, reward))

        # if the next state is a final state then break the loop else update the next state to the current
        # state
        if done:
            break

        state = next_state

    return episode


def generate_policy(Q):
    policy = defaultdict(int)

    states = {state for (state, action) in Q.keys()}
    for state in states:
        policy[state] = max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])
    return policy


def test_policy(policy, env):

    num_episodes = 10000
    num_timesteps = 10000
    total_reward = 0

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(num_timesteps):
            # action = env.action_space.sample()
            action = policy[state]
            next_state, reward, done, info, _ = env.step(action)
            episode_reward += reward

            if done:
                break

            state = next_state

        total_reward += episode_reward

    return total_reward / num_episodes


if __name__ == '__main__':

    env = gym.make('Blackjack-v1', natural=True, sab=False)
    
    Q = defaultdict(float)
    total_return = defaultdict(float)

    N = defaultdict(int)

    num_iterations = 2000
    for i in tqdm.tqdm(range(num_iterations)):
        epsilon = 1 - i / num_iterations
        episode = generate_episode(100, Q)

        # get all the state-action pairs in the episode
        all_state_action_pairs = [(s, a) for (s, a, r) in episode]

        # store all the rewards obtained in the episode in the rewards list
        rewards = [r for (s, a, r) in episode]

        # for each step in the episode
        for t, (state, action, reward) in enumerate(episode):

            # if the state-action pair is occurring for the first time in the episode
            if not (state, action) in all_state_action_pairs[0:t]:
                # compute the return R of the state-action pair as the sum of rewards
                R = sum(rewards[t:])

                # update total return of the state-action pair
                total_return[(state, action)] = total_return[(state, action)] + R

                # update the number of times the state-action pair is visited
                N[(state, action)] += 1

                # compute the Q value by just taking the average
                Q[(state, action)] = total_return[(state, action)] / N[(state, action)]

    policy = generate_policy(Q)
    # print(policy)

    print(test_policy(policy, env))