import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class GraphEnv(gym.Env):

    def __init__(self, N=10):
        super(GraphEnv, self).__init__()
        self.N = N
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=N, shape=(2,), dtype=np.float32)
        
    def sample_actions(self, curr):
        if curr == 1:
            return [self.N, (curr * 2) % (self.N + 1)]
        return [curr - 1, (curr * 2) % (self.N + 1)]
    
    def step(self, action):
        possible_actions = self.sample_actions(self.current_state)
        next_state = possible_actions[action]
        self.current_state = next_state
        
        reward = -1
        done = False
        
        if self.current_state == self.goal:
            reward = 100.0
            done = True
        elif self.steps >= 100:
            done = True

        self.steps += 1
        
        observation = np.array([self.current_state, self.goal], dtype=np.float32)
        return observation, reward, done, False, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        self.current_state = np.random.randint(1, self.N + 1)
        self.goal = np.random.randint(1, self.N + 1)
        while self.current_state == self.goal:
            self.goal = np.random.randint(1, self.N + 1)
        self.steps = 0
        observation = np.array([self.current_state, self.goal], dtype=np.float32)
        return observation, {}

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPONetwork, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        

        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared(x)
        return self.policy(shared_features), self.value(shared_features)

class PPOAgent:
    def __init__(self, env, hidden_dim=64):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = PPONetwork(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        

        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy, _ = self.network(state)
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def compute_advantages(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        advantages = []
        returns = []
        running_return = 0
        previous_value = 0
        running_advantage = 0
        
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                running_return = 0
                running_advantage = 0
                
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
            
            delta = reward + gamma * previous_value * (1 - done) - value
            running_advantage = delta + gamma * lambda_ * running_advantage * (1 - done)
            advantages.insert(0, running_advantage)
            
            previous_value = value
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        

        policy, values = self.network(states)
        values = values.squeeze()
        

        advantages, returns = self.compute_advantages(rewards, values.detach().cpu().numpy(), dones)
        
        # PPO update
        for _ in range(10):

            policy, values = self.network(states)
            values = values.squeeze()
            

            dist = Categorical(policy)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            

            ratio = (new_log_probs - old_log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            

            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            value_loss = 0.5 * (returns - values).pow(2).mean()
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
        return policy_loss.item(), value_loss.item()

def train(num_episodes=1000):
    env = GraphEnv(N=100)
    agent = PPOAgent(env)
    
    episode_rewards = []
    success_rate = []
    batch_size = 32
    success_temp = []
    reward_temp = []
    
    # Training loop
    for episode in range(num_episodes):
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        episode_reward = 0
        state, _ = env.reset()
        done = False
        
        while not done:

            action, log_prob = agent.get_action(state)
            
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            episode_reward += reward

        if len(states) >= batch_size:
            policy_loss, value_loss = agent.update(states, actions, log_probs, rewards, dones)
            states, actions, log_probs, rewards, dones = [], [], [], [], []
            print(f"Episode {episode+1}, Policy Loss: {policy_loss:.2f}, Value Loss: {value_loss:.2f}")
        
        reward_temp.append(episode_reward)
        success = any(r == 100.0 for r in rewards)
        success_temp.append(float(success))
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_temp)
            avg_success = np.mean(success_temp) * 100

            success_temp = []
            reward_temp = []

            episode_rewards.append(avg_reward)
            success_rate.append(avg_success)

            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}, Success Rate: {avg_success:.2f}%")
    
    return episode_rewards, success_rate

if __name__ == "__main__":
    rewards, success_rates = train(num_episodes=10000)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(success_rates)
    plt.title("Success Rate (%)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    
    plt.tight_layout()
    plt.show()