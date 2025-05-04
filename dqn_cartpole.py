#!/usr/bin/env python3
# dqn_cartpole.py
# Deep Q-Network (DQN) implementation to solve CartPole-v1.
# Required dependencies: pip install gymnasium torch numpy matplotlib

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check if GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda (GPU)")
else:
    device = torch.device("cpu")
    print("Using device: cpu (GPU not available)")

# Hyperparameters (Optimized with Optuna)
EPISODES = 500
LEARNING_RATE = 0.0003036075374135476  # Optimized
GAMMA = 0.9516147918855113  # Optimized
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9848432382431861  # Optimized
BATCH_SIZE = 128  # Optimized
REPLAY_BUFFER_SIZE = 10000  # Optimized
TARGET_UPDATE = 13  # Optimized
HIDDEN_SIZES = (128, 256)  # Optimized

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Using optimized architecture with two hidden layers of 256 units each
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.fc3 = nn.Linear(HIDDEN_SIZES[1], output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize two networks: policy network and target network
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        self.epsilon = EPSILON_START
        
    def select_action(self, state, training=True):
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute max Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
    
    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    scores = []
    best_avg_score = -np.inf  # Initialize best average score
    best_model_path = "dqn_cartpole_best.pth" # Path to save the best model
    
    print("Starting training for", EPISODES, "episodes...")
    
    for episode in range(EPISODES):
        state, _ = env.reset(seed=RANDOM_SEED + episode)
        score = 0
        done = False
        
        while not done:
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store the transition in memory
            agent.memory.add(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            score += reward
            
            # Perform one step of the optimization
            agent.learn()
            
        # Update the target network every TARGET_UPDATE episodes
        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()
            
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Average of last 100 episodes
        
        # Save the model if it has the best average score so far
        if len(scores) >= 100 and avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save_model(best_model_path)
            print(f"*** New best model saved with Avg Score: {best_avg_score:.2f} at episode {episode} ***")

        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # We consider the environment solved if average score is 195 over 100 consecutive episodes
        if avg_score >= 195.0 and len(scores) >= 100:
            print(f"Environment solved in {episode} episodes! Average Score: {avg_score:.2f}")
            # Optional: Save the final solved model separately if needed
            # agent.save_model("dqn_cartpole_solved.pth")
            break
    
    env.close()
    # Return the agent (which might be the last state, not necessarily the best) 
    # and scores. We will load the best model in main.
    return agent, scores, best_model_path 

def visualize_results(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('DQN Training on CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('dqn_cartpole_results.png')
    plt.show()

def evaluate(agent, episodes=10, render=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    for episode in range(episodes):
        state, _ = env.reset(seed=RANDOM_SEED + episode)
        score = 0
        done = False
        step = 0
        
        print(f"Starting evaluation episode {episode+1}/{episodes}")
        
        while not done:
            # Select action with no exploration
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            step += 1
            
            if render:
                env.render()
                time.sleep(0.01)
            
            if step % 50 == 0:
                print(f"Step {step}, Current score: {score}")
        
        print(f"Episode {episode+1} finished with score: {score}")
    
    env.close()

def main():
    # Train the agent
    agent, scores, best_model_path = train()
    
    # Plot and save the results
    visualize_results(scores)

    # Load the best model saved during training for evaluation
    print(f"\nLoading best model from {best_model_path} for evaluation...")
    try:
        agent.load_model(best_model_path)
        print("Best model loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Best model file {best_model_path} not found. Evaluating with the final model state.")

    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    evaluate(agent)

if __name__ == "__main__":
    main()
