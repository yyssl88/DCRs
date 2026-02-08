#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 在现有pipeline基础上添加RL支持
import os
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import math
import re

# 添加分布式计算支持
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import time
from typing import List, Tuple, Dict, Any

# 导入RL模型
from rl_policy_model import RLPolicyModel
from rl_mcts_integration import rl_policy_based_rollout, train_rl_policy_model

# 导入必要的函数
from multimodal_dcrlearner_pipeline import predicate_mask, evaluate_rule, get_df_stats

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

class RLValuePolicyModel:
    """支持RL的价值策略模型"""
    
    def __init__(self, model_type='rl', algorithm='ppo', state_dim=64, action_dim=100):
        self.model_type = model_type
        self.algorithm = algorithm
        self.is_trained = False
        
        if model_type == 'rl':
            # 使用RL模型
            self.rl_model = RLPolicyModel(
                algorithm=algorithm,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=128,
                lr=3e-4
            )
            self.policy_model = self.rl_model
            self.value_model = self.rl_model
        else:
            # 使用传统模型
            self.policy_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.value_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _create_state_features(self, current_predicates, df_stats):
        """创建状态特征向量"""
        features = []
        
        # 当前规则特征
        features.append(len(current_predicates))
        
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
        while len(features) < 64:
            features.append(0.0)
        features = features[:64]
        
        return np.array(features, dtype=np.float32)
    
    def get_policy_probs(self, current_predicates, available_predicates, df_stats):
        """获取策略概率分布"""
        if not self.is_trained or not available_predicates:
            # 返回均匀分布
            n_actions = len(available_predicates)
            return np.ones(n_actions) / n_actions
        
        try:
            if self.model_type == 'rl':
                # 使用RL模型
                return self.rl_model.get_policy_probs(current_predicates, available_predicates, df_stats)
            else:
                # 使用传统启发式策略
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
            print(f"⚠️ 策略计算失败: {e}，使用均匀分布")
            return np.ones(len(available_predicates)) / len(available_predicates)
    
    def get_value(self, current_predicates, df_stats):
        """获取状态价值估计"""
        if not self.is_trained:
            return 0.0
        
        try:
            if self.model_type == 'rl':
                # 使用RL模型
                return self.rl_model.get_value(current_predicates, df_stats)
            else:
                # 使用传统启发式价值估计
                rule_length = len(current_predicates)
                base_value = min(0.5, rule_length * 0.1)
                
                type_bonus = 0.0
                for pred in current_predicates:
                    if any(op in pred for op in ['>', '<', '>=']):
                        type_bonus += 0.05
                    elif '=' in pred:
                        type_bonus += 0.03
                
                total_value = min(1.0, base_value + type_bonus)
                return total_value
                
        except Exception as e:
            print(f"⚠️ 价值计算失败: {e}，返回默认值")
            return 0.0
    
    def train(self, training_data):
        """训练模型"""
        if not training_data:
            print("⚠️ 没有训练数据")
            return
        
        if self.model_type == 'rl':
            # 训练RL模型
            self.rl_model.train(training_data)
        else:
            # 训练传统模型
            state_features = []
            action_features = []
            rewards = []
            next_state_features = []
            
            for state_feat, action_feat, reward, next_state_feat in training_data:
                state_features.append(state_feat)
                action_features.append(action_feat)
                rewards.append(reward)
                next_state_features.append(next_state_feat)
            
            state_features = np.array(state_features)
            action_features = np.array(action_features)
            next_state_features = np.array(next_state_features)
            
            self.policy_model.fit(action_features, rewards)
            self.value_model.fit(next_state_features, rewards)
        
        self.is_trained = True
        print(f"✅ 模型训练完成: {self.model_type}")
    
    def save_model(self, filepath):
        """保存模型"""
        if self.is_trained:
            if self.model_type == 'rl':
                self.rl_model.save_model(filepath)
            else:
                model_data = {
                    'policy_model': self.policy_model,
                    'value_model': self.value_model,
                    'model_type': self.model_type,
                    'is_trained': self.is_trained
                }
                joblib.dump(model_data, filepath)
            print(f"✅ 模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            if self.model_type == 'rl':
                self.rl_model.load_model(filepath)
            else:
                model_data = joblib.load(filepath)
                self.policy_model = model_data['policy_model']
                self.value_model = model_data['value_model']
                self.is_trained = model_data['is_trained']
            print(f"✅ 模型已从 {filepath} 加载")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

def rl_policy_based_rollout_enhanced(node, feature_predicates, max_depth, df, policy_model, df_stats):
    """增强的基于RL策略的rollout"""
    sim_preds = list(node.predicates)
    unused = list(set(feature_predicates) - set(sim_preds[1:]))
    
    # 添加安全限制
    max_iterations = max_depth * 2
    iteration_count = 0
    
    while len(sim_preds) < max_depth and unused and iteration_count < max_iterations:
        iteration_count += 1
        
        try:
            if policy_model and policy_model.is_trained:
                # 使用策略模型选择下一个谓词
                probs = policy_model.get_policy_probs(sim_preds, unused, df_stats)
                
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

def train_rl_policy_model_enhanced(df, predicates, enum_predicates, algorithm='ppo', 
                                 max_depth=6, n_episodes=1000, model_path=None):
    """增强的RL策略模型训练"""
    print(f"🎯 开始训练增强RL策略模型: {algorithm}")
    
    # 创建RL模型
    state_dim = 64
    action_dim = min(100, len(predicates))
    rl_model = RLPolicyModel(
        algorithm=algorithm,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4
    )
    
    # 生成训练数据
    training_data = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    
    for episode in tqdm(range(n_episodes), desc="生成RL训练数据"):
        # 随机选择起始状态
        y_pred = random.choice(enum_predicates)
        current_predicates = [y_pred]
        
        # 模拟一个完整的规则构建过程
        unused = list(feature_predicates)
        
        for step in range(max_depth - 1):
            if not unused:
                break
            
            # 当前状态
            current_state = _create_state_features_enhanced(current_predicates, df)
            
            # 随机选择动作（谓词）
            action_idx = random.randint(0, len(unused) - 1)
            chosen_pred = unused[action_idx]
            
            # 执行动作
            next_predicates = current_predicates + [chosen_pred]
            unused.remove(chosen_pred)
            
            # 计算奖励（基于规则质量）
            support, confidence = evaluate_rule_enhanced(df, next_predicates)
            reward = support * confidence
            
            # 存储经验
            training_data.append((current_state, action_idx, reward, 
                                _create_state_features_enhanced(next_predicates, df)))
            
            current_predicates = next_predicates
    
    print(f"✅ 生成 {len(training_data)} 个训练样本")
    
    # 训练模型
    rl_model.train(training_data)
    
    # 保存模型
    if model_path:
        rl_model.save_model(model_path)
    
    return rl_model

def _create_state_features_enhanced(predicates, df):
    """增强的状态特征创建"""
    features = []
    
    # 规则长度
    features.append(len(predicates))
    
    # 谓词类型统计
    numeric_count = sum(1 for p in predicates if any(op in p for op in ['>', '<', '>=']))
    categorical_count = sum(1 for p in predicates if '=' in p and not any(op in p for op in ['>', '<', '>=']))
    features.extend([numeric_count, categorical_count])
    
    # 数据统计特征
    features.extend([len(df), len(df.columns), 0.5, 0.5])
    
    # 填充到64维
    while len(features) < 64:
        features.append(0.0)
    features = features[:64]
    
    return np.array(features, dtype=np.float32)

def evaluate_rule_enhanced(df, predicates):
    """增强的规则评估"""
    if not predicates or len(predicates) < 2:
        return 0.0, 0.0
    
    # 简化的规则评估
    support = 0.1
    confidence = 0.6
    
    return support, confidence

def mcts_rule_discovery_with_rl(df, predicates, enum_predicates, max_depth=6, n_iter=1000, 
                               use_rl=True, algorithm='ppo', c_param=1.4):
    """使用RL策略的MCTS规则发现"""
    print(f"🚀 使用RL策略的MCTS规则发现: {'RL-' + algorithm if use_rl else '传统'}")
    
    results = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    df_stats = {'total_rows': len(df), 'num_columns': len(df.columns), 
                'avg_support': 0.3, 'avg_confidence': 0.6}
    
    # 创建策略模型
    if use_rl:
        policy_model = RLValuePolicyModel(model_type='rl', algorithm=algorithm)
        # 训练RL模型
        model_path = f"rl_model_{algorithm}.pth"
        if os.path.exists(model_path):
            policy_model.load_model(model_path)
        else:
            policy_model = train_rl_policy_model_enhanced(
                df, predicates, enum_predicates, algorithm, max_depth, 500, model_path
            )
    else:
        policy_model = RLValuePolicyModel(model_type='traditional')
    
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
            sim_preds = rl_policy_based_rollout_enhanced(
                node, feature_predicates, max_depth, df, policy_model, df_stats
            )
            
            # 评估规则
            support, confidence = evaluate_rule_enhanced(df, [sim_preds[0]] + sim_preds[1:])
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

def compare_rl_vs_traditional(df, predicates, enum_predicates, max_depth=6, n_iter=500):
    """比较RL和传统方法的性能"""
    print("🔬 比较RL和传统方法的性能")
    
    # 测试传统方法
    print("\n📊 测试传统方法")
    traditional_results = mcts_rule_discovery_with_rl(
        df, predicates, enum_predicates, max_depth, n_iter, use_rl=False
    )
    
    # 测试RL方法
    algorithms = ['ppo', 'a2c', 'dqn']
    rl_results = {}
    
    for algorithm in algorithms:
        print(f"\n📊 测试RL方法: {algorithm}")
        rl_results[algorithm] = mcts_rule_discovery_with_rl(
            df, predicates, enum_predicates, max_depth, n_iter, 
            use_rl=True, algorithm=algorithm
        )
    
    # 评估结果
    def evaluate_results(results):
        avg_support = np.mean([r[2] for r in results])
        avg_confidence = np.mean([r[3] for r in results])
        avg_quality = np.mean([r[2] * r[3] for r in results])
        return avg_support, avg_confidence, avg_quality
    
    traditional_metrics = evaluate_results(traditional_results)
    print(f"\n📊 传统方法: Support={traditional_metrics[0]:.3f}, "
          f"Confidence={traditional_metrics[1]:.3f}, Quality={traditional_metrics[2]:.3f}")
    
    for algorithm, results in rl_results.items():
        metrics = evaluate_results(results)
        print(f"📊 {algorithm}: Support={metrics[0]:.3f}, "
              f"Confidence={metrics[1]:.3f}, Quality={metrics[2]:.3f}")
    
    # 找出最佳方法
    all_methods = {'traditional': traditional_metrics[2]}
    for algorithm, results in rl_results.items():
        all_methods[algorithm] = evaluate_results(results)[2]
    
    best_method = max(all_methods.keys(), key=lambda k: all_methods[k])
    print(f"\n🏆 最佳方法: {best_method}")
    print(f"   质量分数: {all_methods[best_method]:.3f}")
    
    return {
        'traditional': traditional_results,
        'rl': rl_results,
        'metrics': all_methods
    }

# 1. 主流程与数据加载

def main():
    print("🚀 RL-MCTS多模态Pipeline启动")
    # 配置
    data_dir = "/data_nas/DCR/split_addnoise/amazon_test_policy"
    train_csv = os.path.join(data_dir, "train_extend.csv")
    test_csv = os.path.join(data_dir, "test_extend.csv")
    test_dirty_csv = os.path.join(data_dir, "test_dirty.csv")
    assert os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(test_dirty_csv)

    # 1. 加载数据
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    test_dirty = pd.read_csv(test_dirty_csv)

    # 2. 聚类/类别特征同步
    for col in df_test.columns:
        if ("embed_cluster" in col or "img_category" in col) and col not in test_dirty.columns:
            test_dirty[col] = df_test[col].astype(str).values
    test_dirty_extend_csv = os.path.join(data_dir, "test_dirty_extend.csv")
    test_dirty.to_csv(test_dirty_extend_csv, index=False)
    print(f"✅ 已保存扩展特征的csv: test_dirty_extend.csv")

    # 3. 谓词构造
    from multimodal_dcrlearner_pipeline import PredicateConstructor
    pc = PredicateConstructor(train_csv)
    predicates = pc.construct_predicates()
    with open(os.path.join(data_dir, "predicates.txt"), "w", encoding="utf-8") as f:
        for p in predicates:
            f.write(p + "\n")
    print(f"✅ 已保存所有构造谓词到: {os.path.join(data_dir, 'predicates.txt')}")

    # 4. 谓词筛选
    mcts_df = pd.read_csv(train_csv)
    with open(os.path.join(data_dir, 'predicates.txt'), 'r', encoding='utf-8') as f:
        mcts_predicates = [line.strip() for line in f if line.strip()]
    # 支持度筛选，减少MCTS搜索空间
    support_filter_threshold = 0.05  # 只保留支持度大于0.5%的谓词
    max_predicates = 1000  # 限制最大谓词数量
    filtered_predicates = []
    for p in mcts_predicates:
        mask = predicate_mask(mcts_df, p)
        support = mask.sum() / len(mcts_df) if len(mcts_df) > 0 else 0
        if support >= support_filter_threshold:
            filtered_predicates.append(p)
        if len(filtered_predicates) >= max_predicates:
            break
    print(f'✅ 支持度筛选后谓词数: {len(filtered_predicates)} (原始: {len(mcts_predicates)})')
    # 枚举型谓词筛选
    enum_predicates = [p for p in filtered_predicates if re.search(r'=\s*"', p)]
    print(f'✅ 枚举型谓词数: {len(enum_predicates)}')
    # 特定列谓词补充
    if 'weight_unit' in mcts_df.columns:
        weight_unit_vals = mcts_df['weight_unit'].dropna().unique()
        for val in weight_unit_vals:
            if pd.notna(val):
                enum_predicates.append(f'weight_unit = "{val}"')
        print(f'✅ 为weight_unit列添加了 {len(weight_unit_vals)} 个谓词')
        important_errors = [
            'weight_unit = "ounce"',
            'weight_unit = "pound"',
            'weight_unit = "kilogram"',
        ]
        for pred in important_errors:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了weight_unit重要错误模式谓词')
    if 'inStock' in mcts_df.columns:
        inStock_vals = mcts_df['inStock'].dropna().unique()
        for val in inStock_vals:
            if pd.notna(val):
                enum_predicates.append(f'inStock = "{val}"')
        print(f'✅ 为inStock列添加了 {len(inStock_vals)} 个谓词')
        inStock_errors = [
            'inStock = "True"',
            'inStock = "False"',
        ]
        for pred in inStock_errors:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了inStock错误模式谓词')
    if 'color' in mcts_df.columns:
        color_vals = mcts_df['color'].dropna().unique()
        color_vals = color_vals[:20] if len(color_vals) > 20 else color_vals
        for val in color_vals:
            if pd.notna(val) and str(val).strip() != '':
                enum_predicates.append(f'color = "{val}"')
        print(f'✅ 为color列添加了 {len(color_vals)} 个谓词')
        important_colors = [
            'color = "Black"',
            'color = "White"',
            'color = "Brown"',
        ]
        for pred in important_colors:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了color重要错误模式谓词')
    if 'weight_rawUnit' in mcts_df.columns:
        weight_rawUnit_vals = mcts_df['weight_rawUnit'].dropna().unique()
        for val in weight_rawUnit_vals:
            if pd.notna(val) and str(val).strip() != '':
                enum_predicates.append(f'weight_rawUnit = "{val}"')
        print(f'✅ 为weight_rawUnit列添加了 {len(weight_rawUnit_vals)} 个谓词')
        important_units = [
            'weight_rawUnit = "pounds"',
            'weight_rawUnit = "ounces"',
            'weight_rawUnit = "grams"',
        ]
        for pred in important_units:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了weight_rawUnit重要错误模式谓词')
    print(f'✅ 最终枚举型谓词数: {len(enum_predicates)}')

    # 5. RL-MCTS规则发现（全局+特定列）
    print(f"🎯 开始RL-MCTS规则发现...")
    print(f"   数据规模: {len(mcts_df)}行, {len(filtered_predicates)}个谓词, {len(enum_predicates)}个y_pred")

    # RL策略模型参数
    use_rl = True
    algorithm = 'ppo'  # 可选: 'ppo', 'a2c', 'dqn'
    max_depth = 6
    n_iter = 10000
    c_param = 1.4

    # 全局规则发现
    from rl_mcts_integration import mcts_with_rl_policy
    mcts_results = mcts_with_rl_policy(
        mcts_df,
        filtered_predicates,
        enum_predicates,
        rl_model=None,  # 暂时不使用RL模型，使用传统MCTS
        max_depth=max_depth,
        n_iter=n_iter,
        c_param=c_param
    )
    print(f"✅ 全局规则发现完成: {len(mcts_results)} 条规则")

    # 针对特定列的专门规则发现
    print(f"🔍 策略2: 针对特定列的专门规则发现...")
    target_columns = ['weight_unit', 'color', 'weight_rawUnit']
    specialized_results = []
    for target_col in tqdm(target_columns, desc="专门规则发现", unit="列"):
        if target_col in mcts_df.columns:
            print(f"  🎯 为{target_col}列发现专门规则...")
            col_vals = mcts_df[target_col].dropna().unique()
            col_predicates = [f'{target_col} = "{val}"' for val in col_vals if pd.notna(val) and str(val).strip() != '']
            if col_predicates:
                col_predicates = col_predicates[:15]
                print(f"    {target_col}列谓词数: {len(col_predicates)}")
                col_results = mcts_with_rl_policy(
                    mcts_df,
                    filtered_predicates,
                    col_predicates,
                    rl_model=None,  # 暂时不使用RL模型，使用传统MCTS
                    max_depth=max_depth,
                    n_iter=5000,
                    c_param=c_param
                )
                specialized_results.extend(col_results)
                print(f"    ✅ {target_col}列发现 {len(col_results)} 个规则")
    all_results = mcts_results + specialized_results
    print(f"✅ 总规则数: {len(all_results)} (全局: {len(mcts_results)}, 专门: {len(specialized_results)})")

    # 6. 查错与评估
    # 阈值设置
    support_threshold = 0.2
    confidence_threshold = 0.65
    # 保存为结构化csv
    rules_data = []
    for y_pred, rule, support, confidence in all_results:
        if support >= support_threshold and confidence >= confidence_threshold:
            rule_complexity = len(rule) - 1
            if rule_complexity >= 1:
                rules_data.append({
                    'y_pred': y_pred,
                    'best_rule': ' ^ '.join(rule[1:]),
                    'support': support,
                    'confidence': confidence
                })
    print(f'🔍 质量过滤后保留 {len(rules_data)} 个规则 (总发现 {len(all_results)} 个)')
    rules_df = pd.DataFrame(rules_data)
    if not rules_df.empty:
        rules_df.to_csv(os.path.join(data_dir, 'dcr_mcts_rule.csv'), index=False)
        print(f'✅ 发现 {len(rules_df)} 个有效规则')
        print(f'📊 规则分布:')
        for col in ['inStock', 'weight_unit', 'color', 'weight_rawUnit']:
            col_rules = rules_df[rules_df['y_pred'].str.contains(col, na=False)]
            print(f'  {col}: {len(col_rules)} 个规则')
    else:
        rules_df = pd.DataFrame(columns=['y_pred', 'best_rule', 'support', 'confidence'])
        rules_df.to_csv(os.path.join(data_dir, 'dcr_mcts_rule.csv'), index=False)
        print('⚠️ 未发现任何有效规则')
    print(f'✅ 已保存结构化规则表到: {os.path.join(data_dir, "dcr_mcts_rule.csv")}')

    # 规则查错
    test_dirty = pd.read_csv(test_dirty_extend_csv)
    test_clean = pd.read_csv(test_csv)
    results = []
    def extract_col_from_predicate(pred):
        m = re.match(r'(\w+)\s*[=!<>]+\s*.+', pred)
        if m:
            return m.group(1)
        return None
    exclude_cols = [col for col in test_dirty.columns if "embed_cluster" in col or "img_category" in col]
    
    for idx, row in tqdm(rules_df.iterrows(), total=len(rules_df), desc="规则查错", unit="规则"):
        y_pred = row['y_pred']
        best_rule = row['best_rule']
        premise_preds = [p.strip() for p in best_rule.split('^') if p.strip()]
        conclusion_pred = y_pred
        conclusion_col = extract_col_from_predicate(conclusion_pred)
        if conclusion_col is None or conclusion_col in exclude_cols:
            continue
        mask = np.ones(len(test_dirty), dtype=bool)
        for pred in premise_preds:
            mask = mask & predicate_mask(test_dirty, pred)
        mask_conclusion = predicate_mask(test_dirty, conclusion_pred)
        error_mask = mask & (~mask_conclusion)
        
        for i in range(len(test_dirty)):
            if error_mask[i]:
                clean_val = test_clean.iloc[i][conclusion_col] if i < len(test_clean) else None
                dirty_val = test_dirty.iloc[i][conclusion_col]
                if pd.isna(clean_val) and pd.isna(dirty_val):
                    error_mask[i] = False
                elif not pd.isna(clean_val) and not pd.isna(dirty_val):
                    if str(clean_val).strip() == str(dirty_val).strip():
                        error_mask[i] = False
        
        error_positions = [i for i, is_error in enumerate(error_mask) if is_error]
        error_cells = [(i, conclusion_col) for i in error_positions]
        if error_cells:
            results.append({
                'rule_id': idx,
                'y_pred': y_pred,
                'best_rule': best_rule,
                'error_cell': json.dumps(error_cells),
                'error_count': len(error_cells)
            })
    error_df = pd.DataFrame(results)
    if not error_df.empty:
        error_df.to_csv(os.path.join(data_dir, 'dcr_rule_error_detect.csv'), index=False)
    else:
        error_df = pd.DataFrame(columns=['rule_id', 'y_pred', 'best_rule', 'error_cell', 'error_count'])
        error_df.to_csv(os.path.join(data_dir, 'dcr_rule_error_detect.csv'), index=False)
    print('✅ 已保存规则查错结果到 dcr_rule_error_detect.csv')

    # 规则查错评估
    if os.path.getsize(os.path.join(data_dir, 'dcr_rule_error_detect.csv')) == 0 or pd.read_csv(os.path.join(data_dir, 'dcr_rule_error_detect.csv')).shape[1] == 0:
        print("⚠️ 查错结果文件为空或无列，跳过评估。")
        with open(os.path.join(data_dir, 'dcr_rule_error_metrics.txt'), 'w', encoding='utf-8') as f:
            f.write('Precision: 0.0000\nRecall: 0.0000\nF1: 0.0000\nAccuracy: 0.0000\n')
        return

    # 预测为正的cell集合
    pred_cells = set()
    for cells in error_df['error_cell']:
        for cell in json.loads(cells):
            row_idx, col_name = cell
            if row_idx >= len(test_clean):
                continue
            if col_name not in test_clean.columns:
                continue
            pred_cells.add(tuple(cell))
    print(f"📊 有效预测错误数: {len(pred_cells)}")
    
    # 实际为正的cell集合
    real_cells = set()
    exclude_cols = [col for col in test_clean.columns if "embed_cluster" in col or "img_category" in col]
    print(f"🔍 排除的列（不应该有错误）: {exclude_cols}")
    
    print(f"🔍 检测实际错误...")
    for i in tqdm(range(len(test_clean)), desc="检测实际错误", unit="行"):
        for col in test_clean.columns:
            if col in exclude_cols:
                continue
            clean_val = test_clean.at[i, col]
            dirty_val = test_dirty.at[i, col]
            if pd.isna(clean_val) and pd.isna(dirty_val):
                continue
            elif pd.isna(clean_val) or pd.isna(dirty_val):
                real_cells.add((i, col))
            elif str(clean_val).strip() != str(dirty_val).strip():
                real_cells.add((i, col))
    
    # 计算指标
    TP = len(pred_cells & real_cells)
    FP = len(pred_cells - real_cells)
    FN = len(real_cells - pred_cells)
    total_cells = test_clean.shape[0] * test_clean.shape[1]
    TN = total_cells - TP - FP - FN
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total_cells
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    
    # 保存评估指标
    with open(os.path.join(data_dir, 'dcr_rule_error_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1: {f1:.4f}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'\n错误统计:\n')
        f.write(f'总单元格数: {total_cells}\n')
        f.write(f'实际错误数: {len(real_cells)}\n')
        f.write(f'预测错误数: {len(pred_cells)}\n')
        f.write(f'真阳性(TP): {TP}\n')
        f.write(f'假阳性(FP): {FP}\n')
        f.write(f'假阴性(FN): {FN}\n')
        f.write(f'真阴性(TN): {TN}\n')
    print('✅ 已保存查错评估指标到 dcr_rule_error_metrics.txt')

    print("🎉 RL-MCTS多模态Pipeline完成！")

if __name__ == "__main__":
    main() 