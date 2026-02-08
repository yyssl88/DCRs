#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import pickle
import os
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """采样经验"""
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        # Actor (Policy)
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Critic (Value)
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Actor
        actor_x = F.relu(self.actor_fc1(x))
        actor_x = F.relu(self.actor_fc2(actor_x))
        actor_output = F.softmax(self.actor_fc3(actor_x), dim=-1)
        
        # Critic
        critic_x = F.relu(self.critic_fc1(x))
        critic_x = F.relu(self.critic_fc2(critic_x))
        critic_output = self.critic_fc3(critic_x)
        
        return actor_output, critic_output

class RLPolicyModel:
    """强化学习策略模型"""
    
    def __init__(self, algorithm='ppo', state_dim=64, action_dim=100, 
                 hidden_dim=128, lr=3e-4, device='cuda'):
        self.algorithm = algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        
        # 初始化网络
        self._init_networks()
        
        # 初始化优化器
        self._init_optimizers(lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # 训练参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 软更新参数
        self.batch_size = 64
        
        print(f"✅ RL Policy Model initialized: {algorithm}, device: {self.device}")
    
    def _init_networks(self):
        """初始化网络"""
        if self.algorithm == 'dqn':
            # DQN: 分离的Q网络
            self.q_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_q_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            
        elif self.algorithm == 'ddpg':
            # DDPG: Actor-Critic网络
            self.actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.critic = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
            self.target_actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_critic = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            
        elif self.algorithm == 'ppo':
            # PPO: Actor-Critic网络
            self.actor_critic = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            
        elif self.algorithm == 'a2c':
            # A2C: Actor-Critic网络
            self.actor_critic = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
    
    def _init_optimizers(self, lr):
        """初始化优化器"""
        if self.algorithm == 'dqn':
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            
        elif self.algorithm == 'ddpg':
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
            
        elif self.algorithm in ['ppo', 'a2c']:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
    
    def _create_state_features(self, current_predicates, df_stats):
        """创建状态特征向量"""
        features = []
        
        # 当前规则特征
        features.append(len(current_predicates))  # 规则长度
        
        # 数据统计特征
        features.extend([
            df_stats.get('total_rows', 0),
            df_stats.get('num_columns', 0),
            df_stats.get('avg_support', 0.0),
            df_stats.get('avg_confidence', 0.0)
        ])
        
        # 谓词类型特征
        numeric_count = sum(1 for p in current_predicates if any(op in p for op in ['>', '<', '>=']))
        categorical_count = sum(1 for p in current_predicates if '=' in p and not any(op in p for op in ['>', '<', '>=']))
        features.extend([numeric_count, categorical_count])
        
        # 填充到固定维度
        while len(features) < self.state_dim:
            features.append(0.0)
        features = features[:self.state_dim]  # 截断到固定维度
        
        return np.array(features, dtype=np.float32)
    
    def _create_action_features(self, available_predicates):
        """创建动作特征向量"""
        # 简化：直接使用谓词索引作为动作
        return list(range(len(available_predicates)))
    
    def get_policy_probs(self, current_predicates, available_predicates, df_stats):
        """获取策略概率分布"""
        if not self.is_trained or not available_predicates:
            # 返回均匀分布
            n_actions = len(available_predicates)
            return np.ones(n_actions) / n_actions
        
        try:
            # 创建状态特征
            state = self._create_state_features(current_predicates, df_stats)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 获取策略分布
            with torch.no_grad():
                if self.algorithm == 'dqn':
                    q_values = self.q_network(state_tensor)
                    probs = F.softmax(q_values, dim=-1).cpu().numpy()[0]
                    
                elif self.algorithm == 'ddpg':
                    action_probs = self.actor(state_tensor)
                    probs = action_probs.cpu().numpy()[0]
                    
                elif self.algorithm in ['ppo', 'a2c']:
                    action_probs, _ = self.actor_critic(state_tensor)
                    probs = action_probs.cpu().numpy()[0]
                
                # 确保概率分布有效
                probs = np.clip(probs, 1e-8, 1.0)
                probs = probs / np.sum(probs)
                
                # 如果动作数量不匹配，调整概率分布
                if len(probs) != len(available_predicates):
                    # 使用启发式策略作为后备
                    probs = np.ones(len(available_predicates))
                    for i, pred in enumerate(available_predicates):
                        if any(op in pred for op in ['>', '<', '>=']):
                            probs[i] *= 1.2
                        elif '=' in pred:
                            probs[i] *= 1.0
                        else:
                            probs[i] *= 0.8
                    probs = probs / np.sum(probs)
                
                return probs
                
        except Exception as e:
            print(f"⚠️ RL策略计算失败: {e}，使用均匀分布")
            return np.ones(len(available_predicates)) / len(available_predicates)
    
    def get_value(self, current_predicates, df_stats):
        """获取状态价值估计"""
        if not self.is_trained:
            return 0.0
        
        try:
            state = self._create_state_features(current_predicates, df_stats)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.algorithm == 'dqn':
                    q_values = self.q_network(state_tensor)
                    value = torch.max(q_values).cpu().numpy()
                    
                elif self.algorithm == 'ddpg':
                    value = self.critic(state_tensor).cpu().numpy()[0, 0]
                    
                elif self.algorithm in ['ppo', 'a2c']:
                    _, value = self.actor_critic(state_tensor)
                    value = value.cpu().numpy()[0, 0]
                
                return float(value)
                
        except Exception as e:
            print(f"⚠️ RL价值计算失败: {e}，返回默认值")
            return 0.0
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if self.algorithm == 'dqn':
            self._train_dqn(states, actions, rewards, next_states, dones)
        elif self.algorithm == 'ddpg':
            self._train_ddpg(states, actions, rewards, next_states, dones)
        elif self.algorithm == 'ppo':
            self._train_ppo(states, actions, rewards, next_states, dones)
        elif self.algorithm == 'a2c':
            self._train_a2c(states, actions, rewards, next_states, dones)
    
    def _train_dqn(self, states, actions, rewards, next_states, dones):
        """训练DQN"""
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.q_network, self.target_q_network)
    
    def _train_ddpg(self, states, actions, rewards, next_states, dones):
        """训练DDPG"""
        # 训练Critic
        next_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states)
        target_q_values = rewards + (self.gamma * target_q_values * ~dones.unsqueeze(1))
        
        current_q_values = self.critic(states)
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 训练Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
    
    def _train_ppo(self, states, actions, rewards, next_states, dones):
        """训练PPO"""
        # 计算优势函数
        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
            advantages = rewards + (self.gamma * next_values.squeeze() * ~dones) - values.squeeze()
        
        # 计算策略损失
        action_probs, _ = self.actor_critic(states)
        action_probs = action_probs.gather(1, actions.unsqueeze(1))
        
        # 简化的PPO损失
        policy_loss = -(torch.log(action_probs) * advantages.unsqueeze(1)).mean()
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def _train_a2c(self, states, actions, rewards, next_states, dones):
        """训练A2C"""
        # 计算优势函数
        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
            advantages = rewards + (self.gamma * next_values.squeeze() * ~dones) - values.squeeze()
        
        # 计算策略损失
        action_probs, _ = self.actor_critic(states)
        action_probs = action_probs.gather(1, actions.unsqueeze(1))
        
        policy_loss = -(torch.log(action_probs) * advantages.unsqueeze(1)).mean()
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def _soft_update(self, source, target):
        """软更新目标网络"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def train(self, training_data):
        """训练模型"""
        if not training_data:
            print("⚠️ 没有训练数据")
            return
        
        print(f"🎯 开始训练RL策略模型: {self.algorithm}")
        
        # 将训练数据转换为经验
        for state_feat, action_feat, reward, next_state_feat in training_data:
            # 简化：使用动作索引
            action_idx = 0  # 可以根据实际情况调整
            done = False
            
            self.store_experience(state_feat, action_idx, reward, next_state_feat, done)
        
        # 训练多个epoch
        n_epochs = 100
        for epoch in range(n_epochs):
            for _ in range(10):  # 每个epoch训练10步
                self.train_step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}/{n_epochs}")
        
        self.is_trained = True
        print(f"✅ RL策略模型训练完成: {self.algorithm}")
    
    def save_model(self, filepath):
        """保存模型"""
        if self.is_trained:
            model_data = {
                'algorithm': self.algorithm,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'is_trained': self.is_trained,
                'replay_buffer': self.replay_buffer
            }
            
            # 保存网络参数
            if self.algorithm == 'dqn':
                model_data['q_network'] = self.q_network.state_dict()
                model_data['target_q_network'] = self.target_q_network.state_dict()
            elif self.algorithm == 'ddpg':
                model_data['actor'] = self.actor.state_dict()
                model_data['critic'] = self.critic.state_dict()
            elif self.algorithm in ['ppo', 'a2c']:
                model_data['actor_critic'] = self.actor_critic.state_dict()
            
            torch.save(model_data, filepath)
            print(f"✅ RL模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            self.algorithm = model_data['algorithm']
            self.state_dim = model_data['state_dim']
            self.action_dim = model_data['action_dim']
            self.is_trained = model_data['is_trained']
            self.replay_buffer = model_data['replay_buffer']
            
            # 加载网络参数
            if self.algorithm == 'dqn':
                self.q_network.load_state_dict(model_data['q_network'])
                self.target_q_network.load_state_dict(model_data['target_q_network'])
            elif self.algorithm == 'ddpg':
                self.actor.load_state_dict(model_data['actor'])
                self.critic.load_state_dict(model_data['critic'])
            elif self.algorithm in ['ppo', 'a2c']:
                self.actor_critic.load_state_dict(model_data['actor_critic'])
            
            print(f"✅ RL模型已从 {filepath} 加载")
        except Exception as e:
            print(f"❌ RL模型加载失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建RL策略模型
    rl_policy = RLPolicyModel(algorithm='ppo', state_dim=64, action_dim=100)
    
    # 模拟训练数据
    training_data = []
    for i in range(1000):
        state = np.random.randn(64).astype(np.float32)
        action = np.random.randn(64).astype(np.float32)
        reward = np.random.random()
        next_state = np.random.randn(64).astype(np.float32)
        training_data.append((state, action, reward, next_state))
    
    # 训练模型
    rl_policy.train(training_data)
    
    # 测试策略
    test_predicates = ['genre = "Fiction"', 'rating > 4']
    test_available = ['language = "English"', 'format = "Paperback"', 'publisher = "Penguin"']
    test_stats = {'total_rows': 1000, 'num_columns': 10, 'avg_support': 0.3, 'avg_confidence': 0.7}
    
    probs = rl_policy.get_policy_probs(test_predicates, test_available, test_stats)
    value = rl_policy.get_value(test_predicates, test_stats)
    
    print(f"策略概率: {probs}")
    print(f"状态价值: {value}") 