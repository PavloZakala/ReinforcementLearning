import gymnasium as gym
import numpy as np

import time

# https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python/blob/master/03.%20Bellman%20Equation%20and%20Dynamic%20Programming/3.08.%20Solving%20the%20Frozen%20Lake%20Problem%20with%20Policy%20Iteration.ipynb

def show_random_agent(env):
    state = env.reset()

    for t in range(1000):
        env.render()
        state, reward, done, _ = env.step(np.random.randint(0, 4))
        if done:
            break
        time.sleep(0.05)

    env.close()


def compute_value_function(policy, env, num_iterations=1000, threshold=1e-20, gamma=1.0):
    # now, we will initialize the value table, with the value of all states to zero
    value_table = np.zeros(env.observation_space.n)

    # for every iteration
    for i in range(num_iterations):

        # update the value table, that is, we learned that on every iteration, we use the updated value
        # table (state values) from the previous iteration
        updated_value_table = np.copy(value_table)

        # thus, for each state, we select the action according to the given policy and then we update the
        # value of the state using the selected action as shown below

        # for each state
        for s in range(env.observation_space.n):
            # select the action in the state according to the policy
            a = policy[s]

            # compute the value of the state using the selected action
            value_table[s] = sum([prob * (r + gamma * updated_value_table[s_])
                                  for prob, s_, r, _ in env.unwrapped.P[s][a]])

        # after computing the value table, that is, value of all the states, we check whether the
        # difference between value table obtained in the current iteration and previous iteration is
        # less than or equal to a threshold value if it is less then we break the loop and return the
        # value table as an accurate value function of the given policy

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break

    return value_table


def extract_policy(value_table, env, gamma=1.0):

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

def policy_iteration(env):
    # set the number of iterations
    num_iterations = 1000

    # we learned that in the policy iteration method, we begin by initializing a random policy.
    # so, we will initialize the random policy which selects the action 0 in all the states
    # policy = np.zeros(env.observation_space.n)
    policy = np.random.randint(0, env.action_space.n, size=(env.observation_space.n))

    # for every iteration
    for i in range(num_iterations):
        # compute the value function using the policy
        value_function = compute_value_function(policy, env)

        # extract the new policy from the computed value function
        new_policy = extract_policy(value_function, env)

        # if the policy and new_policy are same then break the loop
        if (np.all(policy == new_policy)):
            break

        # else, update the current policy to new_policy
        policy = new_policy

    return policy

def test(env, optimal_policy, render=True):

    state, prob = env.reset()
    if render:
        env.render()

    total_reward = 0
    for _ in range(1000):
        action = int(optimal_policy[state])
        state, reward, done, info, prob = env.step(action)

        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward

if __name__ == '__main__':
    # show_random_agent()

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    optimal_policy = policy_iteration(env)
    print(optimal_policy)

    total_reward = test(env, optimal_policy)
    print(total_reward)

    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, optimal_policy, render=False)
        sum_reward += total_reward

    print(sum_reward / 5000)


    env.close()