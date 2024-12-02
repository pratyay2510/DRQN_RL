import torch
import torch.nn as nn
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import random
import copy
import torchvision.transforms as transforms
import torchvision.models as models
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from IPython.display import Image
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import warnings
warnings.filterwarnings("ignore")





SIZE = (210, 160, 3)
WINDOW_SIZE = 1             # Number of frames to stack together as input to the network
ACTIONS = 6                 # Action space of the environment (6 for assault-v5, 4 for breakout-v5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
print(device)



#Wrapper for environment manipulation and for creating the POMDP formulation
class Wrapper(gym.ActionWrapper):
    """
    A wrapper for creating a partially observable environment by randomly masking frames.

    Args:
        env (gym.Env): The environment to wrap.
        mask_prob (float): Probability of masking a frame (default is 0.01).

    Methods:
        reset(): Resets the environment and applies masking.
        step(action): Takes an action in the environment, applies masking to the resulting frame.
        _apply_mask(observation): Applies a mask to the observation with probability `mask_prob`.
    """
    def __init__(self, env, mask_prob=0.01):
        super(Wrapper, self).__init__(env)
        self.mask_prob = mask_prob

    def reset(self):
        # Reset the environment and return the first observation
        observation = self.env.reset()
        return self._apply_mask(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Apply the mask to make the environment partially observable
        observation = self._apply_mask(observation)
        return observation, reward, done, info

    def _apply_mask(self, observation):
        # Replace the frame with an all-black frame with probability `mask_prob`
        if np.random.rand() < self.mask_prob:
            return np.zeros_like(observation)  # Black frame (all zeros)
        return observation

#LSTM model
class LSTM(nn.Module):
    """
    LSTM-based Q-Network for reinforcement learning.

    Args:
        inp (int): Input size (number of features per timestep).
        hidden (int): Number of hidden units in the LSTM.
        layers (int): Number of layers in the LSTM.

    Methods:
        forward(x): Processes input sequences and returns Q-values for actions.
    """
    def __init__(self, inp, hidden, layers):
        super().__init__()
        self.hidden = 512
        self.layers = layers 
        self.lstm = nn.LSTM(inp, hidden, layers,batch_first=True) 
        self.fc = nn.Linear(hidden, ACTIONS) 

    def forward(self, x):
        self.lstm.flatten_parameters() #Flattening the parameters
        batch_size = x.size(0)
        h0 = torch.zeros(self.layers, batch_size, self.hidden).to(device) #Initializing
        c0 = torch.zeros(self.layers, batch_size, self.hidden).to(device) #Initializing
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) #Taking the final time_step of the time sequence
        return out


#Convolutional feature extractor model
class Net(nn.Module):
    """
    Convolutional feature extractor model for image-based input.

    Layers:
        conv1: First convolutional layer with batch normalization and ReLU activation.
        conv2: Second convolutional layer with batch normalization and ReLU activation.
        conv3: Third convolutional layer with batch normalization and ReLU activation.
        pool: Adaptive average pooling to reduce output dimensions.

    Methods:
        forward(x): Passes input through convolutional layers and pooling, returning a flattened feature vector.
    """
    def __init__(self):
        super().__init__()
        
        # Define three convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        # Pooling layer to reduce dimensions
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Output dimensions match ResNet's feature output (512, 1, 1)

    def forward(self, x):
        # Pass input through the three convolutional layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Global Average Pooling
        x = self.pool(x)
        
        # Flatten the output to match (-1, 512)
        return x.view(-1, 64)

            
#Episodic memory class
class EpisodicMemory:
    """
    Class to manage episodic memory for storing experiences during training.

    Args:
        MAX_LENGTH (int): Maximum number of episodes to store (default is 10).
        vanilla (bool): If True, uses unmodified reward function (default is True).

    Methods:
        start_episode(): Initializes a new episode.
        end_episode(): Ends the current episode and stores it in memory.
        remember(state, next_state, action, reward, done): Adds a step to the current episode.
        sample(episode_size): Samples a batch of episodes from memory.
    """
    def __init__(self, MAX_LENGTH=10, vanilla=True):
        self.memory = []
        self.episode_memory = []
        self.MAX_LENGTH = MAX_LENGTH
        self.vanilla = vanilla

    def start_episode(self):
        self.episode_memory = []

    def end_episode(self):

        self.memory.append(self.episode_memory.copy())
        if len(self.memory) > self.MAX_LENGTH:
            self.memory.pop(0)

    def remember(self, state, next_state, action, reward, done):
        self.episode_memory.append([state, next_state, action, reward, done])

    def sample(self, episode_size):
        batch = []
        if(len(self.memory) == 0):
            return batch
        for _ in range(episode_size):
            index = np.random.randint(0, len(self.memory))
            batch = batch + self.memory[index]
        return batch        
         
            
#Agent class
class Agent:
    """
    Reinforcement learning agent with LSTM and feature extractor.

    Args:
        model (nn.Module): LSTM-based Q-network.
        feature_extractor (nn.Module): Convolutional feature extractor.
        epsMem (EpisodicMemory): Episodic memory for storing experiences.
        batchsize (int): Size of training batches (default is 64).
        ep_window_size (int): Window size for episodic memory (default is 3).
        l_rate (float): Learning rate for optimizer (default is 0.001).
        epsilon_decay (float): Decay rate for exploration (default is 0.9999).

    Methods:
        generate_windows(x, win): Creates sliding windows of input data.
        parse_state(state): Preprocesses image frames (resizes, normalizes).
        make_states(raw_states): Converts raw states into feature-extracted tensors.
        predict(state): Selects an action using epsilon-greedy policy.
        train(): Samples from memory and trains the model.
        train_batch(states, next_states, actions, rewards, dones): Trains the model on a batch of transitions.
    """
    def __init__(self, model, feature_extractor, epsMem, batchsize=64, ep_window_size=3, l_rate=0.001, epsilon_decay=0.9999):
        self.model = model # LSTM based Q Network
        self.feature_extractor = feature_extractor # ResNet18 based Transfer Learned Feature Extractor
        self.target_model = copy.deepcopy(model) # Target Network for Double DQN based learning
        self.epsMem = epsMem
        self.batchsize = batchsize
        self.ep_window_size = ep_window_size
        self.epsilon = 1.0
        self.gamma = 0.9
        
        # optimizer for both feature extractor and model
        self.optimizer = torch.optim.Adagrad(
            list(self.model.parameters()) + 
            list(filter(lambda p: p.requires_grad, self.feature_extractor.parameters())), 
            lr=l_rate)
        
        self.train_count = 0
        self.epsilon_decay = epsilon_decay


    def generate_windows(self, x, win):
        # Optimized function to generate windowed episodic memory
        y = torch.zeros(x.shape[0], win, *x.shape[1:])
        y  = torch.stack([ x[i - win + 1 : i + 1]  for i in range(WINDOW_SIZE-1, x.shape[0]) ])
        return y


    def parse_state(self, state):
        # Normalize the input frames
        normalize = transforms.Normalize(mean=[-0.445 / 0.225] * 3, std=[1 / 0.225] * 3)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            normalize
        ])

        # Convert the input frames to tensors
        torch_images = [preprocess(img) for img in state]
        torch_images = torch.stack(torch_images)
        return torch_images

    def make_states(self, raw_states):
        states = self.parse_state(raw_states)
        states = states.to(device)
        features = self.feature_extractor(states)

        # Generate windowed episodic memory
        result = self.generate_windows(features, WINDOW_SIZE)
        return result

    def predict(self, state):
        state = self.make_states(state)
        pred = self.model(state) # predict the Q values for the current state
        # randomly take an action with epsilon probability
        if random.random() < self.epsilon:
            return random.randint(0, ACTIONS-1)
        else:
            # take the action predicted by the Deep Q network
            return torch.argmax(pred[0]).item()

    def train(self):
        eps_batch = self.epsMem.sample(self.ep_window_size) # sample a batch from the episodic memory
        loss_acc = 0.0
        for i in range(0, len(eps_batch), self.batchsize):
            # generate the states, next_states, actions, rewards and dones for the batch
            raw_batch = eps_batch[(max(i-WINDOW_SIZE+1,0)):(i+self.batchsize)]
            states = [x[0] for x in raw_batch]
            next_states = [x[1] for x in raw_batch]
            if i == 0:
                # if the batch is the first batch, pad the states and next_states with blank frames
                null_paddings = [np.zeros(SIZE, dtype=np.uint8) for i in range(WINDOW_SIZE-1)]
                states = null_paddings + states
                next_states = null_paddings[:-1] + [states[0]] + next_states
            else:
                # if the batch is not the first batch, pad the states and next_states with the last few frames from the previous batch
                start_batch = eps_batch[(i-WINDOW_SIZE+1):i]
                states_padding = [x[0] for x in start_batch]
                next_states_padding = [x[1] for x in start_batch]
                states = states_padding + states
                next_states = next_states_padding + next_states
                
            # Collect actions, rewards, and dones for the batch
            actions = [int(x[2]) for x in raw_batch]
            rewards = [float(x[3]) for x in raw_batch]
            dones = [float(int(x[4])) for x in raw_batch]
                
            batch_length = min(len(states), len(next_states), len(actions), len(rewards), len(dones))
            states = states[:batch_length]
            next_states = next_states[:batch_length]
            actions = actions[:batch_length]
            rewards = rewards[:batch_length]
            dones = dones[:batch_length]

            loss_acc += self.train_batch(states, next_states, actions, rewards, dones)
        return loss_acc


    def train_batch(self, states, next_states, actions, rewards, dones):
        # Train the model using the batch
        self.feature_extractor.train()
        self.feature_extractor.zero_grad()
        states = self.make_states(states)
        next_states = self.make_states(next_states)
        future_reward = torch.max(self.target_model(next_states), 1)[0]
        # future_reward = future_reward.view(-1)
        dones = torch.Tensor(dones).to(device)
        rewards = torch.Tensor(rewards).to(device)
        actions = torch.Tensor(actions).long().to(device)


        # Discount the future reward by gamma
        final_reward = (rewards + future_reward * (1.0 - dones) * self.gamma).detach()
        self.model.train()
        self.model.zero_grad()
        # Predict the reward for the current state
        predicted_reward = self.model(states)
    
        actions_one_hot = torch.nn.functional.one_hot(actions, ACTIONS) #change this to 6 for bowling for assaultv-5
        # Multiply the predicted reward with the one hot encoded actions
        predicted_reward = torch.sum(predicted_reward * actions_one_hot, axis=1)
        # Calculate the loss wrt the final reward
        loss = torch.nn.functional.mse_loss(predicted_reward, final_reward)
        loss.backward() # backpropagate the loss
        self.optimizer.step() # update the weights

        self.train_count += 1

        if self.train_count % 10 == 0:
            # update the target network every 10 iterations
            self.target_model = copy.deepcopy(self.model)

        return loss.item()
    

#Helper functions
def visualize_frames(frames):
    """
    Visualizes a sequence of frames.

    Args:
        frames (list): List of frames to display.
    """
    # Visualize the frames
    plt.imshow(np.hstack(frames))
    plt.axis('off')
    plt.show()

def visualize_rewards(rew):
    """
    Plots the reward over time.

    Args:
        rew (list): List of rewards to plot.
    """
    ax = plt.figure(figsize=(6, 3))
    plt.plot(rew)
    plt.xlabel("Time step in Frames")
    plt.ylabel("Reward")
    plt.show()

def save_frames_as_gif(frames, path='./gifs', filename='gym_animation.gif'):
    """
    Saves a sequence of frames as a GIF.

    Args:
        frames (list): List of frames to save.
        path (str): Directory to save the GIF (default is './gifs').
        filename (str): Name of the GIF file (default is 'gym_animation.gif').
    """
    # function to save the frames as a gif
    plt.figure(figsize=(frames[0].shape[1] / 36.0, frames[0].shape[0] / 36.0), dpi=36)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    # Use the 'imageio' library to save the animation as a gif
    gif_path = path + filename
    writer = imageio.get_writer(gif_path, duration=0.005)
    for i in range(len(frames)):
        writer.append_data(frames[i])
    writer.close()

    plt.cla()
    plt.clf()
    plt.close()

def initialize_model(eps_mem_size=5, num_lstm_hidden_layers=512, batch_size=64, ep_window_size=5, l_rate=0.0005, epsilon_decay=0.99, vanilla = False):
    """
    Initializes the episodic memory, feature extractor, and reinforcement learning agent.

    Args:
        eps_mem_size (int): Size of the episodic memory (default is 5).
        num_lstm_hidden_layers (int): Number of LSTM hidden layers (default is 512).
        batch_size (int): Batch size for training (default is 64).
        ep_window_size (int): Window size for episodic memory (default is 5).
        l_rate (float): Learning rate for optimizer (default is 0.0005).
        epsilon_decay (float): Decay rate for exploration (default is 0.99).
        vanilla (bool): If True, uses unmodified reward function (default is False).

    Returns:
        tuple: Initialized episodic memory, model, and agent.
    """

    epMem = EpisodicMemory(MAX_LENGTH=eps_mem_size, vanilla = vanilla)
    feature_extractor = Net().to(device)
    model = LSTM(64, num_lstm_hidden_layers, 1).to(device)
    agent = Agent(model, feature_extractor, epMem, batch_size, ep_window_size, l_rate, epsilon_decay)
    return (epMem, model, agent)
          
def train(env, agent, epMem, epsilon_decay, num_of_episodes, time_step_size, window_size, show_gifs=False, gif_show_frequency=1, csvfile_name='plot_data.csv'):

  
  scheduler = StepLR(agent.optimizer, step_size=25, gamma=0.8)
  with open(csvfile_name, 'w') as file:
      pass

    
  step_count = 0
  for i in range(1,num_of_episodes-1):
      if(i<200):
            agent.epsilon=1/max(1, i/10)
      else:
           agent.epsilon = 0.05*(epsilon_decay**(i-199))

      # Updated reset logic for compatibility
      reset_result = env.reset()
      if isinstance(reset_result, tuple):  # Newer Gym versions
          observation, info = reset_result
      else:  # Older Gym versions
          observation = reset_result
          info = {}

      # observation, info = env.reset()
      frames = []
      obs, rew = [], []
      curr_state = [np.zeros(SIZE, dtype=np.uint8) for i in range(window_size)]
      total_reward = 0
      epMem.start_episode()
      avg_train_loss = 0
      last_life = time_step_size

      for t in range(time_step_size):
          frames.append(env.render(mode='rgb_array'))
          action = agent.predict(curr_state)
          observation, reward, done, info = env.step(action)
          prev_state = curr_state.copy()
          curr_state.pop(0)
          curr_state.append(observation.copy())
          obs.append(observation.copy())
          rew.append(reward)
          total_reward += reward
          step_count += 1

          epMem.remember(prev_state[-1].copy(), curr_state[-1].copy(), action, reward, done)
          if(t % 100 == 0):
            avg_train_loss += agent.train()
          if done:
              last_life = t
              break

      scheduler.step()
      epMem.end_episode()
      print(f'Episode: {i}, Number of steps: {step_count}, Total Reward: {total_reward}')
      with open(csvfile_name, 'a') as file:
            # log the data to the csv file for plotting subsequently
            x = [total_reward, avg_train_loss/time_step_size, agent.epsilon, last_life, scheduler.get_lr()]
            x = [str(i) for i in x]
            file.write(','.join(x) + '\n')

      if(show_gifs and i % gif_show_frequency == 0):
          # save the frames as a gif if show_gifs is True and at a frequency of gif_show_frequency
          save_frames_as_gif(frames, filename=f'gym_animation_{i//gif_show_frequency}.gif')
          display(Image(data=open(f'gym_animation_{i//gif_show_frequency}.gif','rb').read(), format='png'))       
            
