from collections import deque
import random
import time

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Policy(nn.Module):
    def __init__(self, s_size=6, h_size=32, a_size=3):
        super(Policy, self).__init__()
        self.a_size = a_size

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.a_size)
        q_values = self(torch.tensor(state, dtype=torch.float32, device=device)).to("cpu")
        return torch.argmax(q_values).item()

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = Policy(s_size=state_dim, a_size=action_dim).to(device)
        self.target_model = Policy(s_size=state_dim, a_size=action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        i = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        i += 1
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32, device=device))).item()

            with torch.no_grad():
                target_f = self.target_model(torch.tensor(state, dtype=torch.float32, device=device)).to("cpu").numpy()
                target_f[action] = target
                target_f = torch.tensor(target_f, dtype=torch.float32, device=device)

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.tensor(state, dtype=torch.float32, device=device)))
            loss.backward()
            self.optimizer.step()
        
            if i % 1 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=1000)

    scores = []
    scores_deque = deque(maxlen=100)

    best_reward = -np.inf

    for i_episode in range(200):
        state, _ = env.reset()
        score = 0
        for t in range(1000):
            action = agent.model.act(state, epsilon=agent.epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            score += reward

            agent.replay(16)

            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        print(f'Episode {i_episode} Score {score} Average Score {np.mean(scores_deque)}')

        if i_episode % 30 == 29:
            reward = test(env, agent.model, render=False, num_episodes=5)

            if reward > best_reward:
                best_reward = reward
                torch.save(agent.model.state_dict(), f'checkpoint_{env.spec.id}_best.pth')

def test(env, policy, render=True, num_episodes=1):

    total_reward = 0
    with torch.no_grad():
        for _ in range(num_episodes):

            state, _ = env.reset()
            for _ in range(1000):
                action = policy.act(state)
                state, reward, done, info, _ = env.step(action)

                if render:
                    env.render()

                total_reward += reward
                if done:
                    break

    print(f'Total Reward: {total_reward / num_episodes}')

    return total_reward / num_episodes

if __name__ == '__main__':

    env = gym.make('Acrobot-v1')

    # train(env)

    policy = Policy(s_size=6, a_size=3)
    policy.load_state_dict(torch.load(f'checkpoint_{env.spec.id}.pth'))

    test(env, policy, render=False, num_episodes=1000)
    env = gym.make('Acrobot-v1', render_mode="human")
    test(env, policy, render=True)
