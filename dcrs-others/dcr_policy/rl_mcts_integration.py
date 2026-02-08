#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import random
import os
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from rl_policy_model import RLPolicyModel

# 导入真实的evaluate_rule函数
from multimodal_dcrlearner_pipeline import evaluate_rule

def create_rl_training_data(df, predicates, enum_predicates, max_depth=6, n_episodes=1000):
    """创建强化学习训练数据"""
    print(f"🎯 创建RL训练数据: {n_episodes}个episode")
    
    training_data = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    
    for episode in tqdm(range(n_episodes), desc="生成RL训练数据"):
        # 随机选择起始状态
        y_pred = random.choice(enum_predicates)
        current_predicates = [y_pred]
        
        # 模拟一个完整的规则构建过程
        episode_data = []
        unused = list(feature_predicates)
        
        for step in range(max_depth - 1):
            if not unused:
                break
            
            # 当前状态
            current_state = _create_state_features(current_predicates, df)
            
            # 随机选择动作（谓词）
            action_idx = random.randint(0, len(unused) - 1)
            chosen_pred = unused[action_idx]
            
            # 执行动作
            next_predicates = current_predicates + [chosen_pred]
            unused.remove(chosen_pred)
            
            # 计算奖励（基于规则质量）
            support, confidence = evaluate_rule(df, next_predicates)
            reward = support * confidence
            
            # 存储经验
            episode_data.append({
                'state': current_state,
                'action': action_idx,
                'reward': reward,
                'next_state': _create_state_features(next_predicates, df),
                'done': len(next_predicates) >= max_depth
            })
            
            current_predicates = next_predicates
        
        # 将episode数据添加到训练数据
        training_data.extend(episode_data)
    
    print(f"✅ 生成 {len(training_data)} 个训练样本")
    return training_data

def _create_state_features(predicates, df):
    """创建状态特征向量"""
    features = []
    
    # 规则长度
    features.append(len(predicates))
    
    # 谓词类型统计
    numeric_count = sum(1 for p in predicates if any(op in p for op in ['>', '<', '>=']))
    categorical_count = sum(1 for p in predicates if '=' in p and not any(op in p for op in ['>', '<', '>=']))
    features.extend([numeric_count, categorical_count])
    
    # 数据统计特征（简化）
    features.extend([len(df), len(df.columns), 0.5, 0.5])  # 固定值
    
    # 填充到64维
    while len(features) < 64:
        features.append(0.0)
    features = features[:64]
    
    return np.array(features, dtype=np.float32)

# def evaluate_rule(df, predicates):
#     """评估规则质量"""
#     if not predicates or len(predicates) < 2:
#         return 0.0, 0.0
#     
#     # 简化的规则评估
#     premise_preds = predicates[:-1]
#     conclusion_pred = predicates[-1]
#     
#     # 计算支持度和置信度
#     support = 0.1  # 简化计算
#     confidence = 0.6  # 简化计算
#     
#     return support, confidence

def train_rl_policy_model(df, predicates, enum_predicates, algorithm='ppo', 
                         max_depth=6, n_episodes=1000, model_path=None):
    """训练强化学习策略模型"""
    print(f"🎯 开始训练RL策略模型: {algorithm}")
    
    # 创建RL模型
    state_dim = 64
    action_dim = min(100, len(predicates))  # 限制动作空间
    rl_model = RLPolicyModel(
        algorithm=algorithm,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4
    )
    
    # 生成训练数据
    training_data = create_rl_training_data(
        df, predicates, enum_predicates, max_depth, n_episodes
    )
    
    # 训练模型
    rl_model.train(training_data)
    
    # 保存模型
    if model_path:
        rl_model.save_model(model_path)
    
    return rl_model

def rl_policy_based_rollout(node, feature_predicates, max_depth, df, rl_model, df_stats):
    """基于RL策略的rollout"""
    sim_preds = list(node.predicates)
    unused = list(set(feature_predicates) - set(sim_preds[1:]))
    
    # 添加安全限制
    max_iterations = max_depth * 2
    iteration_count = 0
    
    while len(sim_preds) < max_depth and unused and iteration_count < max_iterations:
        iteration_count += 1
        
        try:
            if rl_model and rl_model.is_trained:
                # 使用RL策略选择下一个谓词
                probs = rl_model.get_policy_probs(sim_preds, unused, df_stats)
                
                # 添加探索性
                epsilon = 0.1  # 探索率
                if random.random() < epsilon:
                    # 随机探索
                    chosen_idx = random.randint(0, len(unused) - 1)
                else:
                    # 利用策略
                    chosen_idx = np.random.choice(len(unused), p=probs)
                
                chosen_pred = unused[chosen_idx]
            else:
                # 随机选择
                chosen_pred = random.choice(unused)
            
            sim_preds.append(chosen_pred)
            unused = list(set(feature_predicates) - set(sim_preds[1:]))
            
        except Exception as e:
            print(f"⚠️ RL Rollout失败: {e}，使用随机选择")
            chosen_pred = random.choice(unused)
            sim_preds.append(chosen_pred)
            unused = list(set(feature_predicates) - set(sim_preds[1:]))
    
    return sim_preds

def mcts_with_rl_policy(df, predicates, enum_predicates, rl_model=None, 
                       max_depth=6, n_iter=1000, c_param=1.4):
    """使用RL策略的MCTS规则发现"""
    if rl_model is not None:
        print(f"🚀 使用RL策略的MCTS规则发现: {rl_model.algorithm}")
    else:
        print(f"🚀 使用传统MCTS规则发现")
    
    results = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    df_stats = {'total_rows': len(df), 'num_columns': len(df.columns), 
                'avg_support': 0.3, 'avg_confidence': 0.6}
    
    for y_pred in tqdm(enum_predicates, desc="RL-MCTS规则发现"):
        root = MCTSNode([y_pred])
        best_support, best_confidence = 0, 0.0
        best_rule = []
        
        for _ in range(n_iter):
            node = root
            # Selection: 使用UCB
            while node.children:
                node = node.best_child(c_param)
                if node is None:
                    break
            
            # Expansion: 随机选择
            if not node.is_terminal(max_depth, feature_predicates):
                node.expand(feature_predicates)
                if node.children:
                    node = random.choice(node.children)
            
            # Simulation: 使用RL策略
            sim_preds = rl_policy_based_rollout(
                node, feature_predicates, max_depth, df, rl_model, df_stats
            )
            
            # 评估规则
            support, confidence = evaluate_rule(df, [sim_preds[0]] + sim_preds[1:])
            reward = support * confidence
            
            # Backpropagation
            tmp_node = node
            while tmp_node:
                tmp_node.visits += 1
                tmp_node.value += reward
                tmp_node = tmp_node.parent
            
            if reward > best_support * best_confidence:
                best_support, best_confidence = support, confidence
                best_rule = list(sim_preds)
        
        results.append((y_pred, best_rule, best_support, best_confidence))
    
    return results

class MCTSNode:
    """MCTS节点类"""
    def __init__(self, predicates, parent=None):
        self.predicates = predicates
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def is_terminal(self, max_depth, all_predicates):
        return len(self.predicates) >= max_depth or len(set(all_predicates) - set(self.predicates)) == 0
    
    def expand(self, all_predicates):
        unused = list(set(all_predicates) - set(self.predicates))
        for p in unused:
            child = MCTSNode(self.predicates + [p], parent=self)
            self.children.append(child)
        return self.children
    
    def best_child(self, c_param=1.4):
        if not self.children:
            return None
        
        # UCB选择
        choices_weights = []
        for child in self.children:
            exploitation = child.value / (child.visits + 1e-6)
            exploration = c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            ucb_score = exploitation + exploration
            choices_weights.append(ucb_score)
        
        return self.children[np.argmax(choices_weights)]

def compare_rl_algorithms(df, predicates, enum_predicates, max_depth=6, n_iter=500):
    """比较不同RL算法的性能"""
    print("🔬 比较不同RL算法的性能")
    
    algorithms = ['ppo', 'a2c', 'dqn']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n📊 测试算法: {algorithm}")
        
        # 训练模型
        model_path = f"rl_model_{algorithm}.pth"
        rl_model = train_rl_policy_model(
            df, predicates, enum_predicates, algorithm, 
            max_depth, n_episodes=500, model_path=model_path
        )
        
        # 运行MCTS
        mcts_results = mcts_with_rl_policy(
            df, predicates, enum_predicates, rl_model, 
            max_depth, n_iter
        )
        
        # 评估结果
        avg_support = np.mean([r[2] for r in mcts_results])
        avg_confidence = np.mean([r[3] for r in mcts_results])
        avg_quality = np.mean([r[2] * r[3] for r in mcts_results])
        
        results[algorithm] = {
            'avg_support': avg_support,
            'avg_confidence': avg_confidence,
            'avg_quality': avg_quality,
            'rules_count': len(mcts_results)
        }
        
        print(f"  {algorithm}: Support={avg_support:.3f}, Confidence={avg_confidence:.3f}, Quality={avg_quality:.3f}")
    
    # 找出最佳算法
    best_algorithm = max(results.keys(), key=lambda k: results[k]['avg_quality'])
    print(f"\n🏆 最佳算法: {best_algorithm}")
    print(f"   质量分数: {results[best_algorithm]['avg_quality']:.3f}")
    
    return results

def online_rl_training(df, predicates, enum_predicates, rl_model, 
                      max_depth=6, n_iter=1000, update_frequency=100):
    """在线RL训练"""
    print(f"🔄 在线RL训练: 每{update_frequency}步更新一次")
    
    results = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    df_stats = {'total_rows': len(df), 'num_columns': len(df.columns), 
                'avg_support': 0.3, 'avg_confidence': 0.6}
    
    for y_pred in tqdm(enum_predicates, desc="在线RL训练"):
        root = MCTSNode([y_pred])
        best_support, best_confidence = 0, 0.0
        best_rule = []
        
        for step in range(n_iter):
            node = root
            # Selection
            while node.children:
                node = node.best_child()
                if node is None:
                    break
            
            # Expansion
            if not node.is_terminal(max_depth, feature_predicates):
                node.expand(feature_predicates)
                if node.children:
                    node = random.choice(node.children)
            
            # Simulation with RL
            sim_preds = rl_policy_based_rollout(
                node, feature_predicates, max_depth, df, rl_model, df_stats
            )
            
            # 评估并存储经验
            support, confidence = evaluate_rule(df, [sim_preds[0]] + sim_preds[1:])
            reward = support * confidence
            
            # 存储经验用于在线学习
            if rl_model and rl_model.is_trained:
                current_state = _create_state_features(node.predicates, df)
                next_state = _create_state_features(sim_preds, df)
                rl_model.store_experience(current_state, 0, reward, next_state, False)
            
            # Backpropagation
            tmp_node = node
            while tmp_node:
                tmp_node.visits += 1
                tmp_node.value += reward
                tmp_node = tmp_node.parent
            
            # 在线更新RL模型
            if step % update_frequency == 0 and rl_model and rl_model.is_trained:
                rl_model.train_step()
            
            if reward > best_support * best_confidence:
                best_support, best_confidence = support, confidence
                best_rule = list(sim_preds)
        
        results.append((y_pred, best_rule, best_support, best_confidence))
    
    return results

# 使用示例
if __name__ == "__main__":
    # 模拟数据
    import pandas as pd
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'genre': np.random.choice(['Fiction', 'Non-Fiction', 'Mystery'], n_samples),
        'rating': np.random.randint(1, 6, n_samples),
        'language': np.random.choice(['English', 'Spanish', 'French'], n_samples),
        'format': np.random.choice(['Paperback', 'Hardcover', 'Ebook'], n_samples),
        'publisher': np.random.choice(['Penguin', 'Random House', 'HarperCollins'], n_samples)
    })
    
    # 创建谓词
    predicates = [
        'genre = "Fiction"', 'genre = "Non-Fiction"', 'genre = "Mystery"',
        'rating > 3', 'rating > 4', 'rating = 5',
        'language = "English"', 'language = "Spanish"', 'language = "French"',
        'format = "Paperback"', 'format = "Hardcover"', 'format = "Ebook"',
        'publisher = "Penguin"', 'publisher = "Random House"', 'publisher = "HarperCollins"'
    ]
    
    enum_predicates = ['genre = "Fiction"', 'genre = "Non-Fiction"', 'rating > 4']
    
    print("🚀 RL-MCTS集成测试")
    
    # 1. 比较不同RL算法
    results = compare_rl_algorithms(df, predicates, enum_predicates)
    
    # 2. 使用最佳算法进行在线训练
    best_algorithm = max(results.keys(), key=lambda k: results[k]['avg_quality'])
    rl_model = RLPolicyModel(algorithm=best_algorithm, state_dim=64, action_dim=100)
    
    online_results = online_rl_training(df, predicates, enum_predicates, rl_model)
    
    print(f"\n✅ 在线训练完成，发现 {len(online_results)} 个规则")
    avg_quality = np.mean([r[2] * r[3] for r in online_results])
    print(f"   平均质量: {avg_quality:.3f}") 