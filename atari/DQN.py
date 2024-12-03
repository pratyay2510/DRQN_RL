import gym
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import random
import torch.optim as optim
import torch.optim as optim
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blackout_probability = 0.5 # Put this as 0 for non flickering environment

#DQN Network
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Initialize the DQN network.
        
        :param input_shape: Tuple representing the input shape (channels, height, width).
        :param num_actions: Number of possible actions (output Q-values for each action).
        """
        super(DQN, self).__init__()
        
        # First convolutional layer: 32 filters of size 8x8 with stride 4
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        
        # Second convolutional layer: 64 filters of size 4x4 with stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        
        # Third convolutional layer: 64 filters of size 3x3 with stride 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Calculate the flattened size after convolutions
        def conv_output_size(size, kernel_size, stride, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1

        conv_h = conv_output_size(
            conv_output_size(conv_output_size(input_shape[1], 8, 4), 4, 2), 3, 1
        )
        conv_w = conv_output_size(
            conv_output_size(conv_output_size(input_shape[2], 8, 4), 4, 2), 3, 1
        )
        linear_input_size = conv_h * conv_w * 64

        # Fully connected layer for output Q-values
        self.fc = nn.Linear(linear_input_size, num_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Tensor of shape (batch_size, num_actions) representing Q-values.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return x
    

#Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        :param capacity: Maximum number of transitions the buffer can hold.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # Tracks the next position to overwrite when the buffer is full

    def store(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        :param state: Current state (preprocessed).
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state (preprocessed).
        :param done: Whether the episode is done.
        """
        # Create a tuple for the transition
        transition = (state, action, reward, next_state, done)

        # If the buffer isn't full, add the transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Overwrite the oldest transition
            self.buffer[self.position] = transition

        # Update the position to overwrite
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
    
        return random.sample(self.buffer, batch_size)
    
    

    def size(self):

        return len(self.buffer)


#Agent
class DQNAgent:
    def __init__(self, env, input_shape, num_actions, batch_size=32, gamma=0.99, lr=0.0001, target_update_frequency=1000, n_frames=1):
        """
        Initialize the DQN agent.

        :param env: The Gym environment.
        :param replay_buffer: ReplayBuffer instance for experience storage.
        :param input_shape: Shape of the input state (e.g., (1, 84, 84)).
        :param num_actions: Number of possible actions in the environment.
        :param batch_size: Batch size for training.
        :param gamma: Discount factor for future rewards.
        :param lr: Learning rate for the optimizer.
        :param target_update_frequency: Steps between target network updates.
        """
        self.env = env
        self.replay_buffer = ReplayBuffer(capacity=50)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.num_actions = num_actions
        self.n_frames = n_frames

        # Q-network and target network
        self.q_network = DQN(input_shape, num_actions).to(device)
        self.target_network = DQN(input_shape, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize with same weights
        self.target_network.eval()  # Target network doesn't train

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Epsilon-greedy parameters
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Training step counter
        self.step_count = 0
        
        # Initialize the frame stack
        self.frame_stack = []
        
    
    def reset_frame_stack(self):
        """Reset the frame stack."""
        initial_frame = preprocess_observation(self.env.reset()) 
        initial_frame = flickering(initial_frame, blackout_probability)
        self.frame_stack = [initial_frame] * self.n_frames  # Duplicate the first frame

    def stack_frames(self, new_frame):
        """Update the frame stack with a new frame."""
        self.frame_stack.pop(0)  # Remove the oldest frame
        self.frame_stack.append(new_frame)  # Add the new frame
        
        return np.stack(self.frame_stack, axis=0)

    def select_action(self, stacked_state):
        """
        Select an action using epsilon-greedy policy.

        :param state: Current state of the environment.
        :return: Chosen action.
        """
        
        #state = self.stack_frames(state)
        
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            state_tensor = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0).to(device)
            #state_tensor = state_tensor.permute(0, 3, 1, 2) 
            #print("after permute state_tensor", state_tensor.size())
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit: max Q-value action
        

    def train_step(self):
        """
        Perform one training step.
        """
        if self.replay_buffer.size() < self.batch_size:
            return  # Not enough data to train
        
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Sample a batch from the replay buffer
        #single_transition = replay_buffer.sample(batch_size=1)[0]  
        #state, action, reward, next_state, done = single_transition

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        #print("Initial state print before flickering ", states.shape)
        #apply flickering
        states = torch.stack([flickering(frame, blackout_probability) for frame in states])
        next_states = torch.stack([flickering(frame, blackout_probability) for frame in next_states])

        #print("Initial state print after flicker", states.shape)
        
        # Compute current Q-values
        #print("Initial state print before permutation ", states.shape)
        #states = states.permute(0, 3, 1, 2)  
        #print("Initial state print after permutation ", states.shape)
        
        q_values = self.q_network(states)
        #print("Q_values size", q_values.size())
        #actions = actions.unsqueeze(0) 
        #print("action size", actions.size())
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        
        #formating for the network
        #print("next state before permutation ", next_states.size())

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and backpropagate
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


#Training
def train_dqn(agent, num_episodes, max_steps_per_episode, csv_filename):
    """
    Train the DQN agent.

    :param agent: The DQNAgent instance.
    :param num_episodes: Number of episodes to train.
    :param max_steps_per_episode: Maximum steps per episode.
    """
    
    # Initialize the CSV file with a header
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_reward", "loss", "epsilon"])  # Write the header row
        
        
    for episode in range(num_episodes):
        
        # Start the stacked frames
        agent.reset_frame_stack()
        agent_reset= agent.env.reset()
        #print(agent_reset.shape)
        initial_frame = preprocess_observation(agent_reset) # Preprocess the initial observation
        initial_frame = flickering(initial_frame, blackout_probability)
        #print("Initial frame ", initial_frame.shape)
        state = agent.stack_frames(initial_frame)  
        
        
        total_reward = 0
        loss = 0

        for step in range(max_steps_per_episode):
            # Select an action using the epsilon-greedy policy
            action = agent.select_action(state)
            
            # Take the action in the environment
            next_frame, reward, done, _ = agent.env.step(action)
            next_frame = preprocess_observation(next_frame)
            next_frame = flickering(next_frame, blackout_probability)
            #print("frame ", next_frame.shape)

            # Update the frame stack with the new frame
            next_state = agent.stack_frames(next_frame)
            #print(next_state.shape)

            # Store the transition in the replay buffer
            agent.replay_buffer.store(state, action, reward, next_state, done)

            # Train the network
            loss = agent.train_step()

            # Update total reward
            total_reward += reward

            # Update the current state
            state = next_state

            # Break the loop if the episode is done
            if done:
                break

        # Decay epsilon
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        
         # Save results to CSV
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode + 1, total_reward, loss, agent.epsilon])  # Append the row

        print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")

        
    print(f"Training results saved to {csv_filename}")


#Evaluation
def evaluate_agent(agent, num_episodes, max_steps_per_episode):
    """
    Evaluate the DQN agent with stacked frames.

    :param agent: The DQNAgent instance.
    :param num_episodes: Number of evaluation episodes.
    :param max_steps_per_episode: Maximum steps per episode.
    """
    agent.epsilon = 0.0  # Turn off exploration
    total_rewards = []

    for episode in range(num_episodes):
        # Reset the environment and initialize the frame stack
        agent.reset_frame_stack()
        initial_frame = preprocess_observation(agent.env.reset())
        state = agent.stack_frames(initial_frame)  # Initialize the stacked state

        total_reward = 0

        for step in range(max_steps_per_episode):
            # Select the best action
            action = agent.select_action(state)

            # Take the action in the environment
            next_frame, reward, done, _ = agent.env.step(action)
            next_frame = preprocess_observation(next_frame)

            # Update the frame stack with the new frame
            state = agent.stack_frames(next_frame)

            # Update the total reward
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}")

    print(f"Average Reward over {num_episodes} Episodes: {np.mean(total_rewards):.2f}")

    



#Helper functions
def flickering(frame, blackout_probability):
    """
    With a given probability, replaces the frame with a black canvas or leaves it unchanged.

    :param frame: Input frame, either a NumPy array or a PyTorch tensor.
    :param blackout_probability: Probability of replacing the frame with a black canvas.
    :return: Modified frame (same type as input).
    """
    if np.random.rand() < blackout_probability:
        # Handle PyTorch tensors
        if isinstance(frame, torch.Tensor):
            result = torch.zeros_like(frame)
            return result
        
        
        else:
            result = np.zeros_like(frame)
            return result
    
    else:
        return frame

def preprocess_observation(obs):
    """
    Preprocesses an Atari observation from (210, 160, 3) to (84, 84, 1) grayscale.

    :param obs: Raw observation from the environment (NumPy array of shape (210, 160, 3)).
    :return: Preprocessed observation (NumPy array of shape (84, 84, 1)).
    """
    # Convert to grayscale
    gray_obs = rgb2gray(obs)  # Shape: (210, 160)
    
    # Resize to 84x84
    resized_obs = resize(gray_obs, (84, 84), anti_aliasing=True)  # Shape: (84, 84)
    
    # Normalize pixel values to [0, 1]
    normalized_obs = resized_obs / 255.0
    
    # Add a channel dimension to make it (84, 84, 1)
    #preprocessed_obs = np.expand_dims(normalized_obs, axis=-1)
    
    return normalized_obs