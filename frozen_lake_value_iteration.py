import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import time

# https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python/blob/master/03.%20Bellman%20Equation%20and%20Dynamic%20Programming/3.06.%20Solving%20the%20Frozen%20Lake%20Problem%20with%20Value%20Iteration.ipynb
def show_random_agent(env):
    state = env.reset()
    env.render()

    for t in range(1000):

        state, reward, done, _ = env.step(np.random.randint(0, 4))
        env.render()
        if done:
            break
        time.sleep(0.05)

    env.close()


# value iteration method
def value_iteration(env, num_iterations=1000, threshold=1e-20, gamma=0.99):

    # now, we will initialize the value table, with the value of all states to zero
    value_table = np.zeros(env.observation_space.n)

    # for every iteration
    for i in range(num_iterations):

        # update the value table, that is, we learned that on every iteration, we use the updated value
        # table (state values) from the previous iteration
        updated_value_table = np.copy(value_table)

        # now, we compute the value function (state value) by taking the maximum of Q value.

        # thus, for each state, we compute the Q values of all the actions in the state and then
        # we update the value of the state as the one which has maximum Q value as shown below:
        for s in range(env.observation_space.n):
            Q_values = [sum([prob * (r + gamma * updated_value_table[s_])
                             for prob, s_, r, _ in env.unwrapped.P[s][a]])
                        for a in range(env.action_space.n)]

            value_table[s] = max(Q_values)

        print(value_table[0:4])
        print(value_table[4:8])
        print(value_table[8:12])
        print(value_table[12:16])
        print("***************************")

        # after computing the value table, that is, value of all the states, we check whether the
        # difference between value table obtained in the current iteration and previous iteration is
        # less than or equal to a threshold value if it is less then we break the loop and return the
        # value table as our optimal value function as shown below:

        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            break

    return value_table

def extract_policy(env, value_table):
    # set the discount factor
    gamma = 1.0

    # 0: Move left
    # 1: Move down
    # 2: Move right
    # 3: Move up

    # first, we initialize the policy with zeros, that is, first, we set the actions for all the states to
    # be zero
    policy = np.zeros(env.observation_space.n)

    # now, we compute the Q function using the optimal value function obtained from the
    # previous step. After computing the Q function, we can extract policy by selecting action which has
    # maximum Q value. Since we are computing the Q function using the optimal value
    # function, the policy extracted from the Q function will be the optimal policy.

    # As shown below, for each state, we compute the Q values for all the actions in the state and
    # then we extract policy by selecting the action which has maximum Q value.

    # for each state
    for s in range(env.observation_space.n):
        # compute the Q value of all the actions in the state
        Q_values = [sum([prob * (r + gamma * value_table[s_])
                         for prob, s_, r, _ in env.unwrapped.P[s][a]])
                    for a in range(env.action_space.n)]

        # extract policy by selecting the action which has maximum Q value
        policy[s] = np.argmax(np.array(Q_values))

    return policy


def test(env, optimal_policy, render=True):

    state = env.reset()[0]
    if render:
        env.render()


    total_reward = 0
    counter = 0
    for _ in range(1000):
        action = int(optimal_policy[state])
        state, reward, done, info, prob = env.step(action)

        if render:
            env.render()

        total_reward += reward
        counter += 1
        if done:
            break

    return total_reward, counter


if __name__ == '__main__':
    desc=None
    # desc=generate_random_map(size=8)
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)
    # env.seed(42)

    # show_random_agent(env)
    # env.P[1][1][2] = (1 / 3, 2, 1.0, False)
    # env.P[1][2][1] = (1/3, 2, 1.0, False)
    # env.P[1][3][0] = (1 / 3, 2, 1.0, False)
    #
    # env.P[3][0][1] = (1/3, 2, 1.0, False)
    # env.P[3][1][0] = (1 / 3, 2, 1.0, False)
    # env.P[3][3][2] = (1 / 3, 2, 1.0, False)
    #
    # env.P[6][0][0] = (1/3, 2, 1.0, False)
    # env.P[6][2][2] = (1 / 3, 2, 1.0, False)
    # env.P[6][3][1] = (1 / 3, 2, 1.0, False)

    optimal_value_function = value_iteration(env=env)
    optimal_policy = extract_policy(env, optimal_value_function)

    print(optimal_policy)

    total_reward = test(env, optimal_policy)
    print(total_reward)

    sum_reward = 0
    counter_sum = 0
    for _ in range(5000):
        total_reward, counter = test(env, optimal_policy, render=False)
        sum_reward += total_reward
        if sum_reward > 0:
            counter_sum += counter


    print(sum_reward / 5000)
    print(counter_sum / sum_reward)

    env.close()
