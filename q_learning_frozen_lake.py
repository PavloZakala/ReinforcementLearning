import gymnasium as gym
import random
import numpy as np
import tqdm

from collections import defaultdict


def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])

def generate_policy(Q):

    policy = defaultdict(int)

    for state in range(env.observation_space.n):
        policy[state] = max(list(range(env.action_space.n)), key = lambda x: Q[(state, x)])

    return policy

# Off-policy Q-learning
def q_learning(env, num_episodes, num_timesteps, alpha, gamma, epsilon, epsilon_decay=0.95):
    Q = defaultdict(float)

    for i in tqdm.tqdm(range(num_episodes)):
        state = env.reset()[0]

        for t in range(num_timesteps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            next_action = np.argmax([Q[(next_state, a)] for a in range(env.action_space.n)])

            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state = next_state

            if done:
                break
        epsilon = max(epsilon * epsilon_decay, 0.1)

    return Q

def test(env, optimal_policy, render=True):

    state = env.reset()[0]
    if render:
        env.render()


    total_reward = 0
    for _ in range(1000):
        action = int(optimal_policy[state])
        state, reward, done, info, _ = env.step(action)

        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward

def show_agent(env, policy):
    state = env.reset()[0]
    env.render()

    for t in range(1000):

        state, reward, done, _, _ = env.step(policy[state])
        env.render()
        if done:
            break

    env.close()

def generate_random_policy(env):
    policy = defaultdict(int)

    for state in range(env.observation_space.n):
        policy[state] = env.action_space.sample()

    return policy



if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    alpha = 0.1
    gamma = 0.99
    epsilon = 0.7

    num_episodes = 50000
    num_timesteps = 1000

    Q = q_learning(env, num_episodes, num_timesteps, alpha, gamma, epsilon)

    policy = generate_policy(Q)
    # policy = generate_random_policy(env)

    print(policy)

    show_agent(env, policy)

    sum_reward = 0
    for _ in range(50000):
        total_reward = test(env, policy, render=False)
        sum_reward += total_reward

    print(sum_reward / 50000)

    env.close()

