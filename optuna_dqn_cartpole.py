#!/usr/bin/env python3
# optuna_dqn_cartpole.py
# Hyperparameter optimization for DQN CartPole using Optuna
# Required dependencies: pip install gymnasium torch numpy matplotlib optuna

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
import json  # Import the json library

# --- Constants ---
OPTUNA_TIMEOUT_MINUTES = 30 # Duration for Optuna optimization in minutes

# Check if GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda (GPU)")
else:
    device = torch.device("cpu")
    print("Using device: cpu (GPU not available)")

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

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128)):
        super(DQN, self).__init__()
        
        # Create layers dynamically based on hidden_sizes
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, params):
        self.state_size = state_size
        self.action_size = action_size
        
        # Unpack hyperparameters
        self.learning_rate = params['learning_rate']
        self.gamma = params['gamma']
        self.epsilon_start = params['epsilon_start']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.target_update = params['target_update']
        self.hidden_sizes = params['hidden_sizes']
        self.buffer_size = params['buffer_size']
        
        # Initialize networks
        self.policy_net = DQN(state_size, action_size, self.hidden_sizes).to(device)
        self.target_net = DQN(state_size, action_size, self.hidden_sizes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.buffer_size)
        
        self.epsilon = self.epsilon_start
        
    def select_action(self, state, training=True):
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute max Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
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
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Function to train the agent
def train_agent(params, max_episodes=200, seed=42):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, params)
    scores = []
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.add(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            agent.learn()
            
        if episode % agent.target_update == 0:
            agent.update_target_net()
            
        scores.append(score)
        
        # Early stopping if we solve the environment
        if np.mean(scores[-100:]) >= 195.0 and len(scores) >= 100:
            break
    
    env.close()
    # Return mean score over last 100 episodes, or all if less than 100
    return np.mean(scores[-min(100, len(scores)):])

# Optuna objective function
def objective(trial):
    # Define hyperparameters to optimize
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),
        'epsilon_start': 1.0,  # Fixed
        'epsilon_end': 0.01,   # Fixed
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.97, 0.995),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'target_update': trial.suggest_int('target_update', 5, 20),
        'hidden_layer1': trial.suggest_categorical('hidden_layer1', [64, 128, 256]),
        'hidden_layer2': trial.suggest_categorical('hidden_layer2', [64, 128, 256]),
        'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
    }

    # Construct the hidden_sizes tuple required by the agent
    params['hidden_sizes'] = (params['hidden_layer1'], params['hidden_layer2'])
    
    # Train with these hyperparameters and return mean score
    return train_agent(params, max_episodes=200)

def run_optimization():
    timeout_seconds = OPTUNA_TIMEOUT_MINUTES * 60
    print(f"Starting hyperparameter optimization with Optuna for {OPTUNA_TIMEOUT_MINUTES} minutes ({timeout_seconds} seconds)...")
    
    # Create an Optuna study and optimize with a timeout
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=timeout_seconds)
    
    print("Optimization finished!")
    print(f"Number of trials completed: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (mean reward): {study.best_trial.value}")
    
    best_params = study.best_params
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Save the best hyperparameters to a JSON file
    output_filename = "optuna_dqn_best_params.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Best hyperparameters saved to {output_filename}")
    except Exception as e:
        print(f"Error saving hyperparameters to JSON: {e}")

    return best_params

def train_and_evaluate_best_model(best_params, render=True):
    # Set the fixed parameters
    best_params['epsilon_start'] = 1.0
    best_params['epsilon_end'] = 0.01
    
    # Construct the hidden_sizes tuple from individual layer parameters in best_params
    # This is necessary because study.best_params only includes suggested hyperparameters
    best_params['hidden_sizes'] = (
        best_params['hidden_layer1'],
        best_params['hidden_layer2']
    )
    
    # Training
    print("\nTraining final model with best hyperparameters...")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, best_params)
    scores = []
    
    max_episodes = 500
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.add(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            agent.learn()
            
        if episode % agent.target_update == 0:
            agent.update_target_net()
            
        scores.append(score)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Check if we've solved the environment
        if avg_score >= 195.0 and len(scores) >= 100:
            print(f"Environment solved in {episode} episodes! Average Score: {avg_score:.2f}")
            break
    
    env.close()
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('DQN Training on CartPole-v1 with Optimal Hyperparameters')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('optimized_dqn_cartpole_results.png')
    
    # Evaluate the trained agent
    if render:
        print("\nEvaluating trained agent...")
        env = gym.make("CartPole-v1", render_mode="human")
        
        for eval_episode in range(5):  # Show 5 episodes
            state, _ = env.reset(seed=seed + max_episodes + eval_episode)
            score = 0
            done = False
            step = 0
            
            print(f"Starting evaluation episode {eval_episode+1}/5")
            
            while not done:
                action = agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                score += reward
                step += 1
                
                env.render()
                time.sleep(0.01)
                
                if step % 100 == 0:
                    print(f"Step {step}, Current score: {score}")
            
            print(f"Episode {eval_episode+1} finished with score: {score}")
        
        env.close()
    
    return scores

def main():
    # Run hyperparameter optimization
    best_params = run_optimization()
    
    # Train and evaluate the best model
    train_and_evaluate_best_model(best_params)

if __name__ == "__main__":
    main()
