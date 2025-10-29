import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import tqdm
import gymnasium as gym
from collections import deque
import numpy as np

torch.manual_seed(0)  # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# https://www.gymlibrary.dev/environments/classic_control/acrobot/

class Policy(nn.Module):
    def __init__(self, s_size=6, h_size=32, a_size=3):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item() - 1, m.log_prob(action)


def reinforce(env, policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in tqdm.tqdm(range(1, n_episodes + 1)):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            torch.save(policy.state_dict(), 'checkpoint.pth')
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    scores = reinforce(env, policy=policy, optimizer=optimizer)