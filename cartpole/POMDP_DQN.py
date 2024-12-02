import sys
from typing import Dict, List, Tuple
import csv
import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Q_net(nn.Module):
    def __init__(self, state_space=None, action_space=None, sequence_length=10):
        super(Q_net, self).__init__()
        assert state_space is not None, "state_space must be specified."
        assert action_space is not None, "action_space must be specified."

        self.Linear1 = nn.Linear(state_space * sequence_length, 64)
        self.Linear2 = nn.Linear(64, 64)
        self.Linear3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return self.forward(obs).argmax().item()


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, sequence_length: int, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        # Ensure sufficient population for sampling
        valid_size = max(self.size - self.sequence_length, 1)
        sample_size = min(self.batch_size, valid_size)  # Avoid sampling more than available valid sequences

        idxs = np.random.choice(valid_size, size=sample_size, replace=False)
        obs_sequences = np.array([self.obs_buf[i:i + self.sequence_length] for i in idxs])
        next_obs_sequences = np.array([self.next_obs_buf[i:i + self.sequence_length] for i in idxs])

        return dict(obs=obs_sequences,
                    next_obs=next_obs_sequences,
                    acts=self.acts_buf[idxs + self.sequence_length - 1],
                    rews=self.rews_buf[idxs + self.sequence_length - 1],
                    done=self.done_buf[idxs + self.sequence_length - 1])


    def __len__(self):
        return self.size


class Agent:
    def __init__(self, env_name="CartPole-v1", seed=1, device="cuda", buffer_len=100000, batch_size=8,
                 learning_rate=1e-3, gamma=0.99, tau=1e-2, sequence_length=10):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        # Set seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.sequence_length = sequence_length
        self.state_space = self.env.observation_space.shape[0] - 2

        # Define Q networks
        self.Q = Q_net(state_space=self.state_space, action_space=self.env.action_space.n,
                       sequence_length=sequence_length).to(self.device)
        self.Q_target = Q_net(state_space=self.state_space, action_space=self.env.action_space.n,
                              sequence_length=sequence_length).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.state_space, size=buffer_len, sequence_length=sequence_length,
                                          batch_size=batch_size)

        # Logging
        self.writer = SummaryWriter('runs/' + env_name + "_DQN_POMDP_SEED_" + str(seed))
        self.csv_file = "cartpole_dqn_10.csv"

        # Initialize CSV file for rewards logging
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Mean_Reward"])

    def train(self, episodes=650, max_step=2000, print_per_iter=20, eps_start=0.1, eps_end=0.001, eps_decay=0.995):
        epsilon = eps_start
        scores = []
        mean_rewards = []
        step_count = 0
        state_history = collections.deque(maxlen=self.sequence_length)

        for i in range(episodes):
            s = self.env.reset()
            s = s[::2]  # Use only Position of Cart and Pole
            state_history.extend([s] * self.sequence_length)  # Initialize history with the current state
            score = 0
            done = False

            for t in range(max_step):
                state_sequence = np.array(state_history).flatten()
                a = self.Q.sample_action(torch.from_numpy(state_sequence).float().to(self.device), epsilon)
                s_prime, r, done, _ = self.env.step(a)
                s_prime = s_prime[::2]  # Partial observation
                state_history.append(s_prime)

                done_mask = 0.0 if done else 1.0
                self.replay_buffer.put(state_history[-self.sequence_length], a, r / 100.0,
                                    state_history[-1], done_mask)
                score += r
                step_count += 1

                # Check if replay buffer is ready for sampling
                if len(self.replay_buffer) >= self.batch_size + self.sequence_length - 1:
                    self.update_model()

                    if (t + 1) % print_per_iter == 0:
                        self.soft_update_target()

                if done:
                    break

            epsilon = max(eps_end, epsilon * eps_decay)
            scores.append(score)
            mean_reward = np.mean(scores[-50:])  # Mean reward over the last 50 episodes
            mean_rewards.append(mean_reward)

            # Save rewards and mean rewards to CSV
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i + 1, score, mean_reward])

            self.writer.add_scalar('Rewards per episodes', score, i)
            self.writer.add_scalar('Mean Reward (Last 50)', mean_reward, i)

            if i % print_per_iter == 0 and i != 0:
                print(f"Episode: {i}, Steps: {step_count}, Score: {mean_reward:.1f}")

        self.writer.close()
        self.env.close()

        # Save the model at the end of training
        self.save_model("DQN_POMDP_10.pth")


    def update_model(self):
        samples = self.replay_buffer.sample()

        # Dynamically adjust to the actual batch size returned by the replay buffer
        batch_size = samples["obs"].shape[0]

        states = torch.FloatTensor(samples["obs"].reshape(batch_size, -1)).to(self.device)
        actions = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(samples["next_obs"].reshape(batch_size, -1)).to(self.device)
        dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        q_target_max = self.Q_target(next_states).max(1)[0].unsqueeze(1).detach()
        targets = rewards + self.gamma * q_target_max * dones
        q_out = self.Q(states)
        q_a = q_out.gather(1, actions)

        loss = F.smooth_l1_loss(q_a, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def soft_update_target(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path='default.pth'):
        torch.save(self.Q.state_dict(), path)



