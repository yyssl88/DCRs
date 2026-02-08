#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
# 设置只使用5号GPU卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)


class MultimodalModels:
    """多模态模型集合 - External Models M𝑈"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._init_models()
    
    def _init_models(self):
        """初始化多模态模型"""
        print("🔧 Initializing Multimodal Models M𝑈...")
        
        # 文档类模型
        self.bert_mrc = self._create_bert_mrc()
        
        # 多模态模型
        self.qwen_vl = self._create_qwen_vl()

        
        print("✅ Multimodal Models M𝑈 initialized")
    
    def _create_bert_mrc(self):
        """创建Bert-MRC [43] - 实体提取模型，使用BGE-M3"""
        # 使用本地BGE-M3模型
        model_path = "/data_nas/model_hub/bge-m3"
        
        print(f"📁 Loading BGE-M3 from local path: {model_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.bert_model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        print("✅ BGE-M3 [43] initialized")
        return True
    

    
    def _create_qwen_vl(self):
        """创建Qwen-2.5-VL [73] - 图像/视频处理模型，使用Qwen2.5-VL-7B-Instruct"""
        # 使用本地Qwen2.5-VL-7B-Instruct模型
        model_path = "/data_nas/model_hub/Qwen2.5-VL-7B-Instruct"
        
        print(f"📁 Loading Qwen2.5-VL-7B-Instruct from local path: {model_path}")
        self.qwen_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        
        # 加载支持生成的模型类
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.qwen_model.to(self.device)
        self.qwen_model.eval()
        print("✅ Qwen2.5-VL-7B-Instruct [73] initialized (with generation support)")
        return True
    
    def _load_and_preprocess_image(self, image_path):
        """加载和预处理图像"""
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 定义预处理变换
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ])
            
            # 应用变换
            image_tensor = transform(image)
            
            return image_tensor
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            # 返回一个默认的图像张量
            return torch.zeros(3, 224, 224)
    
    def extract_image_features(self, image_path):
        """仅提取图像特征embedding，保留原始维度"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            print(f"✅ Loaded image: {image.size}")
            with torch.no_grad():
                prompt = "Describe this image."
                inputs = self.qwen_processor(
                    text=prompt,
                    images=image, 
                    return_tensors="pt"
                )
                if 'pixel_values' in inputs:
                    inputs['images'] = inputs.pop('pixel_values')
                model_inputs = {}
                for key in ['input_ids', 'attention_mask', 'pixel_values']:
                    if key in inputs:
                        model_inputs[key] = inputs[key]
                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
                outputs = self.qwen_model(**model_inputs)
                if hasattr(outputs, 'image_embeds'):
                    img_feat = outputs.image_embeds
                elif hasattr(outputs, 'last_hidden_state'):
                    img_feat = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'logits'):
                    img_feat = outputs.logits.mean(dim=1)
                else:
                    img_feat = torch.zeros(1, 1).to(self.device)
                return img_feat
        except Exception as e:
            print(f"❌ Error extracting image features from {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_image_category(self, image_path):
        """仅识别图像类别或所属label"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            print(f"✅ Loaded image: {image.size}")
            with torch.no_grad():
                category_prompt = "请识别图片的主要类别或标签"
                text_inputs = self.qwen_processor(
                    text=category_prompt,
                    images=image,
                    return_tensors="pt"
                )
                if 'pixel_values' in text_inputs:
                    text_inputs['images'] = text_inputs.pop('pixel_values')
                model_text_inputs = {}
                for key in ['input_ids', 'attention_mask', 'pixel_values']:
                    if key in text_inputs:
                        model_text_inputs[key] = text_inputs[key]
                model_text_inputs = {k: v.to(self.device) for k, v in model_text_inputs.items()}
                generated_ids = self.qwen_model.generate(
                    **model_text_inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.qwen_processor.tokenizer.eos_token_id
                )
                category = self.qwen_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                category = category.replace(category_prompt, "").strip()
                if not category or len(category) < 2:
                    category = "未知类别"
                return category
        except Exception as e:
            print(f"❌ Error extracting image category from {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def extract_text_features(self, text):
        """提取单个文本的embedding，保留原始维度"""
        if not text or not isinstance(text, str):
            return torch.zeros(1).to(self.device)
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            text_feat = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
            return text_feat.squeeze(0)
    

class PredicateConstructor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.col_types = self._infer_col_types()
        self.row_ids = [f"t{i}" for i in range(len(self.df))]

    def _infer_col_types(self):
        col_types = {}
        for col in self.df.columns:
            if "img_path" in col:
                col_types[col] = "img_path"
            elif self.df[col].dtype in ['float64', 'int64']:
                col_types[col] = "numeric"
            elif self.df[col].dtype == 'object':
                unique_count = self.df[col].nunique()
                avg_len = self.df[col].astype(str).apply(len).mean()
                if unique_count < 20:
                    col_types[col] = "enum"
                elif avg_len > 20:
                    col_types[col] = "text"
                else:
                    col_types[col] = "enum"
            else:
                col_types[col] = "other"
        return col_types

    def construct_predicates(self):
        predicates = []
        for col, typ in self.col_types.items():
            if typ == "enum":
                try:
                    values = self.df[col].dropna().astype(str)
                    if len(values) == 0:
                        continue
                    # 增加谓词多样性，但保持精确性
                    top_modes = values.value_counts().index[:5]  # 增加更多众数
                    for val in top_modes:
                        predicates.append(f'{col} = "{val}"')

                except Exception as e:
                    print(f"⚠️ 枚举型列 {col} 构造谓词出错: {e}")
            elif typ == "numeric" and str(self.df[col].dtype).startswith("int"):
                try:
                    values = self.df[col].dropna().astype(int)
                    if len(values) == 0:
                        continue
                    mean_val = int(values.mean())
                    min_val = int(values.min())
                    max_val = int(values.max())
                    median_val = int(values.median())
                    q25_val = int(values.quantile(0.25))
                    q75_val = int(values.quantile(0.75))
                    for val in set([mean_val, min_val, max_val, median_val, q25_val, q75_val]):
                        predicates.append(f'{col} = {val}')
                        # predicates.append(f'{col} != {val}')  # 注释掉不等于谓词
                        predicates.append(f'{col} > {val}')   # 注释掉大于谓词
                        predicates.append(f'{col} < {val}')   # 注释掉小于谓词
                except Exception as e:
                    print(f"⚠️ int数值型列 {col} 构造谓词出错: {e}")
            elif typ == "numeric" and str(self.df[col].dtype).startswith("float"):
                try:
                    values = self.df[col].dropna().astype(float)
                    if len(values) == 0:
                        continue
                    mean_val = float(values.mean())
                    min_val = float(values.min())
                    max_val = float(values.max())
                    median_val = float(values.median())
                    q25_val = float(values.quantile(0.25))
                    q75_val = float(values.quantile(0.75))
                    for val in [mean_val, min_val, max_val, median_val, q25_val, q75_val]:
                        predicates.append(f'{col} = {val}')  # 重新启用float列的 = 谓词
                        predicates.append(f'{col} > {val}')   # 注释掉大于谓词
                        predicates.append(f'{col} < {val}')   # 注释掉小于谓词
                except Exception as e:
                    print(f"⚠️ float数值型列 {col} 构造谓词出错: {e}")
            # 为embedding聚类列构造谓词
            elif "embed_cluster" in col or "img_category" in col:
                try:
                    values = self.df[col].dropna()
                    if len(values) == 0:
                        continue
                    unique_vals = values.unique()
                    for val in unique_vals[:10]:  # 限制最多10个聚类
                        if pd.notna(val):
                            predicates.append(f'{col} = {val}')
                except Exception as e:
                    print(f"⚠️ 聚类列 {col} 构造谓词出错: {e}")
        return predicates


class ValuePolicyModel:
    """价值策略模型，用于MCTS的policy-based rollout"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.policy_model = None  # 策略模型：预测选择每个谓词的概率
        self.value_model = None   # 价值模型：预测状态的价值
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _create_feature_vector(self, current_predicates, available_predicates, df_stats):
        """创建特征向量"""
        features = []
        
        # 当前谓词组合的特征
        features.append(len(current_predicates))  # 当前规则长度
        
        # 可用谓词的特征
        features.append(len(available_predicates))  # 可用谓词数量
        
        # 数据统计特征
        features.extend([
            df_stats.get('total_rows', 0),
            df_stats.get('num_columns', 0),
            df_stats.get('avg_support', 0.0),
            df_stats.get('avg_confidence', 0.0)
        ])
        
        # 谓词类型特征（简化版）
        numeric_count = sum(1 for p in available_predicates if any(op in p for op in ['>', '<', '>=']))
        categorical_count = sum(1 for p in available_predicates if '=' in p and not any(op in p for op in ['>', '<', '>=']))
        features.extend([numeric_count, categorical_count])
        
        return np.array(features)
    
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
        
        return np.array(features)
    
    def train(self, training_data):
        """训练策略和价值模型
        
        Args:
            training_data: [(state_features, action_features, reward, next_state_features), ...]
        """
        if not training_data:
            print("⚠️ 没有训练数据，使用随机策略")
            return
            
        # 分离数据
        state_features = []
        action_features = []
        rewards = []
        next_state_features = []
        
        for state_feat, action_feat, reward, next_state_feat in training_data:
            state_features.append(state_feat)
            action_features.append(action_feat)
            rewards.append(reward)
            next_state_features.append(next_state_feat)
        
        # 标准化特征
        state_features = np.array(state_features)
        action_features = np.array(action_features)
        next_state_features = np.array(next_state_features)
        
        # 训练策略模型（回归器）
        if self.model_type == 'random_forest':
            self.policy_model = RandomForestRegressor(n_estimators=100, random_state=42)
            # 使用reward作为目标值，训练一个回归模型来预测动作价值
            if len(action_features) > 0:
                self.policy_model.fit(action_features, rewards)
            else:
                # 如果没有有效数据，创建一个简单的模型
                self.policy_model = RandomForestRegressor(n_estimators=10, random_state=42)
                dummy_features = np.zeros((1, 10))
                dummy_labels = np.array([0.0])
                self.policy_model.fit(dummy_features, dummy_labels)
        
        # 训练价值模型（回归器）
        if self.model_type == 'random_forest':
            self.value_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.value_model.fit(next_state_features, rewards)
        
        self.is_trained = True
        print(f"✅ 策略和价值模型训练完成，使用 {len(training_data)} 个样本")
    
    def get_policy_probs(self, current_predicates, available_predicates, df_stats):
        """获取策略概率分布"""
        if not self.is_trained or not available_predicates:
            # 返回均匀分布
            n_actions = len(available_predicates)
            return np.ones(n_actions) / n_actions
        
        # 简化策略：直接返回均匀分布，避免复杂的特征计算
        # 原来的方法会导致特征重复计算和可能的无限循环
        try:
            # 使用简单的启发式策略
            probs = np.ones(len(available_predicates))
            
            # 根据谓词类型给予不同权重
            for i, pred in enumerate(available_predicates):
                # 数值型谓词权重稍高
                if any(op in pred for op in ['>', '<', '>=']):
                    probs[i] *= 1.2
                # 分类型谓词权重适中
                elif '=' in pred:
                    probs[i] *= 1.0
                # 其他谓词权重稍低
                else:
                    probs[i] *= 0.8
            
            # 归一化
            probs = probs / np.sum(probs)
            return probs
            
        except Exception as e:
            # 如果计算失败，返回均匀分布
            print(f"⚠️ 策略计算失败: {e}，使用均匀分布")
            return np.ones(len(available_predicates)) / len(available_predicates)
    
    def get_value(self, current_predicates, df_stats):
        """获取状态价值估计"""
        if not self.is_trained:
            return 0.0
        
        try:
            # 简化价值估计：基于规则长度和谓词类型
            rule_length = len(current_predicates)
            
            # 基础价值：规则越长价值越高（但有限制）
            base_value = min(0.5, rule_length * 0.1)
            
            # 根据谓词类型调整价值
            type_bonus = 0.0
            for pred in current_predicates:
                if any(op in pred for op in ['>', '<', '>=']):
                    type_bonus += 0.05  # 数值型谓词加分
                elif '=' in pred:
                    type_bonus += 0.03  # 分类型谓词加分
            
            total_value = min(1.0, base_value + type_bonus)
            return total_value
            
        except Exception as e:
            print(f"⚠️ 价值计算失败: {e}，返回默认值")
            return 0.0
    
    def save_model(self, filepath):
        """保存模型"""
        if self.is_trained:
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
            model_data = joblib.load(filepath)
            self.policy_model = model_data['policy_model']
            self.value_model = model_data['value_model']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            print(f"✅ 模型已从 {filepath} 加载")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")


class MCTSNode:
    def __init__(self, predicates, parent=None):
        self.predicates = predicates  # 当前节点的谓词组合（list）
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior_value = 0.0  # 先验价值（来自策略模型）

    def is_terminal(self, max_depth, all_predicates):
        return len(self.predicates) >= max_depth or len(set(all_predicates) - set(self.predicates)) == 0

    def expand(self, all_predicates):
        unused = list(set(all_predicates) - set(self.predicates))
        for p in unused:
            child = MCTSNode(self.predicates + [p], parent=self)
            self.children.append(child)
        return self.children

    def best_child(self, c_param=1.4, alpha=1.0):
        import numpy as np
        if not self.children:
            return None
            
        # 使用UCB进行节点选择，alpha参数影响探索项
        choices_weights = []
        for child in self.children:
            # UCB公式：exploitation + alpha * exploration
            exploitation = child.value / (child.visits + 1e-6)
            exploration = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            ucb_score = exploitation + alpha * exploration
            choices_weights.append(ucb_score)
        
        return self.children[np.argmax(choices_weights)]

# 修改predicate_mask支持col op val格式

def predicate_mask(df, pred):
    import re
    import numpy as np
    m = re.match(r'(\w+)\s*([=!<>]+)\s*(.+)', pred)
    if not m:
        return np.ones(len(df), dtype=bool)
    col, op, val = m.groups()
    val = val.strip('"')
    if col not in df.columns:
        return np.ones(len(df), dtype=bool)
    if op == '=':
        return df[col].astype(str) == val
    elif op == '!=':
        return df[col].astype(str) != val
    elif op == '>':
        return df[col].astype(float) > float(val)
    elif op == '<':
        return df[col].astype(float) < float(val)
    else:
        return np.ones(len(df), dtype=bool)

def evaluate_rule(df, predicates, target_col='y'):
    import numpy as np
    if not predicates or len(predicates) < 2:
        return 0, 0.0
    premise_preds = predicates[:-1]
    conclusion_pred = predicates[-1]
    # 前提mask
    mask = np.ones(len(df), dtype=bool)
    for pred in premise_preds:
        mask = mask & predicate_mask(df, pred)
    support_count = mask.sum()
    support = support_count / len(df) if len(df) > 0 else 0
    if support_count == 0:
        return 0, 0.0
    # 结论mask
    mask_conclusion = mask & predicate_mask(df, conclusion_pred)
    confidence = mask_conclusion.sum() / support_count
    return support, confidence


def get_df_stats(df, predicates):
    """获取数据统计信息，用于策略模型特征"""
    stats = {
        'total_rows': len(df),
        'num_columns': len(df.columns),
        'avg_support': 0.0,
        'avg_confidence': 0.0
    }
    
    # 计算平均支持度和置信度
    total_support = 0.0
    total_confidence = 0.0
    count = 0
    
    for pred in predicates[:min(100, len(predicates))]:  # 限制计算量
        try:
            support, confidence = evaluate_rule(df, [pred])
            total_support += support
            total_confidence += confidence
            count += 1
        except:
            continue
    
    if count > 0:
        stats['avg_support'] = total_support / count
        stats['avg_confidence'] = total_confidence / count
    
    return stats

def policy_based_rollout(node, feature_predicates, max_depth, df, policy_model, df_stats):
    """基于策略的rollout"""
    sim_preds = list(node.predicates)
    unused = list(set(feature_predicates) - set(sim_preds[1:]))
    
    # 添加安全限制，避免无限循环
    max_iterations = max_depth * 2
    iteration_count = 0
    
    while len(sim_preds) < max_depth and unused and iteration_count < max_iterations:
        iteration_count += 1
        
        try:
            if policy_model and policy_model.is_trained:
                # 使用策略模型选择下一个谓词
                probs = policy_model.get_policy_probs(sim_preds, unused, df_stats)
                # 添加一些随机性
                probs = probs * 0.8 + np.random.uniform(0, 0.2, len(probs))
                probs = probs / probs.sum()  # 重新归一化
                
                chosen_idx = np.random.choice(len(unused), p=probs)
                chosen_pred = unused[chosen_idx]
            else:
                # 随机选择
                chosen_pred = random.choice(unused)
            
            sim_preds.append(chosen_pred)
            unused = list(set(feature_predicates) - set(sim_preds[1:]))
            
        except Exception as e:
            # 如果策略选择失败，使用随机选择
            print(f"⚠️ Rollout策略选择失败: {e}，使用随机选择")
            chosen_pred = random.choice(unused)
            sim_preds.append(chosen_pred)
            unused = list(set(feature_predicates) - set(sim_preds[1:]))
    
    return sim_preds


def mcts_rule_discovery(df, predicates, enum_predicates, max_depth=3, n_iter=100, 
                       use_policy=True, policy_model=None, c_param=1.4):
    """增强的MCTS规则发现，支持策略模型"""
    results = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    
    # 获取数据统计信息
    df_stats = get_df_stats(df, predicates)
    
    for y_pred in tqdm(enum_predicates, desc="MCTS规则发现", unit="y_pred"):
        root = MCTSNode([y_pred])
        best_support, best_confidence = 0, 0.0
        best_rule = []
        
        for _ in range(n_iter):
            node = root
            # Selection
            while node.children:
                node = node.best_child(c_param)
                if node is None:
                    break
            
            # Expansion
            if not node.is_terminal(max_depth, feature_predicates):
                node.expand(feature_predicates)
                if node.children:
                    # 为子节点设置先验价值
                    if policy_model and policy_model.is_trained:
                        for child in node.children:
                            child.prior_value = policy_model.get_value(child.predicates, df_stats)
                    
                    # 选择新扩展的节点
                    if use_policy and policy_model and policy_model.is_trained:
                        # 基于策略选择
                        probs = policy_model.get_policy_probs(node.predicates, 
                                                            [c.predicates[-1] for c in node.children], df_stats)
                        node = np.random.choice(node.children, p=probs)
                    else:
                        # 随机选择
                        node = random.choice(node.children)
            
            # Simulation (Policy-based rollout)
            sim_preds = policy_based_rollout(node, feature_predicates, max_depth, df, policy_model, df_stats)
            
            # 评估：前提=sim_preds[1:], 结论=sim_preds[0]
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

def mcts_rule_discovery_single_y_pred(args):
    """单个y_pred的MCTS规则发现（用于并行计算）"""
    if len(args) == 5:
        # 兼容旧版本调用
        df, predicates, y_pred, max_depth, n_iter = args
        use_policy = False
        policy_model = None
        c_param = 1.4
        alpha = 1.0
    elif len(args) == 8:
        # 新版本调用，支持策略模型
        df, predicates, y_pred, max_depth, n_iter, use_policy, policy_model, c_param = args
        alpha = 1.0
    else:
        # 最新版本调用，支持alpha参数
        df, predicates, y_pred, max_depth, n_iter, use_policy, policy_model, c_param, alpha = args
    
    feature_predicates = [p for p in predicates if p not in [y_pred]]
    
    # 获取数据统计信息
    df_stats = get_df_stats(df, predicates)
    
    root = MCTSNode([y_pred])
    best_support, best_confidence = 0, 0.0
    best_rule = []
    
    for _ in range(n_iter):
        node = root
        # Selection
        while node.children:
            node = node.best_child(c_param, alpha)
            if node is None:
                break
        
        # Expansion
        if not node.is_terminal(max_depth, feature_predicates):
            node.expand(feature_predicates)
            if node.children:
                # 为子节点设置先验价值
                if policy_model and policy_model.is_trained:
                    for child in node.children:
                        child.prior_value = policy_model.get_value(child.predicates, df_stats)
                
                # 选择新扩展的节点
                if use_policy and policy_model and policy_model.is_trained:
                    # 基于策略选择
                    probs = policy_model.get_policy_probs(node.predicates, 
                                                        [c.predicates[-1] for c in node.children], df_stats)
                    node = np.random.choice(node.children, p=probs)
                else:
                    # 随机选择
                    node = random.choice(node.children)
        
        # Simulation (Policy-based rollout)
        sim_preds = policy_based_rollout(node, feature_predicates, max_depth, df, policy_model, df_stats)
        
        # 评估：前提=sim_preds[1:], 结论=sim_preds[0]
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
    
    return (y_pred, best_rule, best_support, best_confidence)

def mcts_rule_discovery_yroot(df, predicates, enum_predicates, max_depth=3, n_iter=100, n_workers=None, use_parallel=True, 
                             use_policy=True, policy_model=None, c_param=1.4, alpha=1.0):
    """
    分布式MCTS规则发现（支持策略模型）
    
    Args:
        df: 数据框
        predicates: 所有谓词
        enum_predicates: 枚举型谓词（作为结论）
        max_depth: 最大规则深度
        n_iter: 每个y_pred的迭代次数
        n_workers: 并行工作进程数，None表示使用CPU核心数
        use_policy: 是否使用策略模型
        policy_model: 策略模型实例
        c_param: MCTS探索参数
    """
    if not use_parallel or len(enum_predicates) < 5:
        # 单线程模式（用于调试或小规模数据）
        print(f"🔄 使用单线程MCTS: {len(enum_predicates)}个y_pred")
        if use_policy and policy_model and policy_model.is_trained:
            print(f"🎯 使用策略模型进行rollout")
        start_time = time.time()
        results = []
        for i, y_pred in enumerate(tqdm(enum_predicates, desc="单线程MCTS", unit="y_pred")):
            args = (df, predicates, y_pred, max_depth, n_iter, use_policy, policy_model, c_param, alpha)
            result = mcts_rule_discovery_single_y_pred(args)
            results.append(result)
        return results
    
    # 多线程模式
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(enum_predicates))
    
    print(f"🚀 启动分布式MCTS规则发现: {len(enum_predicates)}个y_pred, {n_workers}个工作进程")
    if use_policy and policy_model and policy_model.is_trained:
        print(f"🎯 使用启发式策略模型进行rollout")
    start_time = time.time()
    
    # 准备参数（使用简化的启发式策略）
    args_list = [(df, predicates, y_pred, max_depth, n_iter, use_policy, policy_model, c_param, alpha) 
                 for y_pred in enum_predicates]
    
    # 使用进程池进行并行计算
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_y_pred = {executor.submit(mcts_rule_discovery_single_y_pred, args): args[2] 
                           for args in args_list}
        
        # 收集结果
        completed = 0
        with tqdm(total=len(enum_predicates), desc="分布式MCTS", unit="y_pred") as pbar:
            for future in as_completed(future_to_y_pred):
                y_pred = future_to_y_pred[future]
                try:
                    # 添加超时机制，避免无限等待
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)
                    completed += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"❌ y_pred {y_pred} 处理失败: {e}")
                    # 添加默认结果
                    results.append((y_pred, [], 0, 0.0))
                    completed += 1
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    print(f"✅ 分布式MCTS完成: {len(results)}个规则, 耗时 {elapsed_time:.2f}秒")
    
    # 打印性能统计
    print_performance_stats(start_time, len(enum_predicates), n_workers)
    
    return results


def get_distributed_config():
    """获取分布式计算配置"""
    config = {
        'n_workers': None,  # None表示使用CPU核心数
        'use_multiprocessing': True,  # 是否使用多进程
        'chunk_size': 10,  # 每个进程处理的y_pred数量
        'max_workers': mp.cpu_count(),  # 最大工作进程数
    }
    
    # 可以根据数据规模调整
    if config['max_workers'] > 8:
        config['n_workers'] = 8  # 限制最大进程数
    else:
        config['n_workers'] = config['max_workers']
    
    return config

def print_performance_stats(start_time, n_y_preds, n_workers):
    """打印性能统计信息"""
    elapsed_time = time.time() - start_time
    avg_time_per_pred = elapsed_time / n_y_preds if n_y_preds > 0 else 0
    theoretical_speedup = n_workers if n_workers > 1 else 1
    
    print(f"📊 性能统计:")
    print(f"   总耗时: {elapsed_time:.2f}秒")
    print(f"   平均每个y_pred耗时: {avg_time_per_pred:.2f}秒")
    print(f"   工作进程数: {n_workers}")
    print(f"   理论加速比: {theoretical_speedup}x")
    print(f"   实际加速比: {elapsed_time / (avg_time_per_pred * n_y_preds / n_workers) if n_workers > 1 else 1:.2f}x")

def train_policy_model(df, predicates, enum_predicates, max_depth=3, n_samples=1000, model_path=None):
    """训练策略模型
    
    Args:
        df: 数据框
        predicates: 所有谓词
        enum_predicates: 枚举型谓词
        max_depth: 最大规则深度
        n_samples: 训练样本数量
        model_path: 模型保存路径
    
    Returns:
        ValuePolicyModel: 训练好的策略模型
    """
    print(f"🎯 开始训练策略模型，生成 {n_samples} 个训练样本...")
    
    policy_model = ValuePolicyModel()
    training_data = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    df_stats = get_df_stats(df, predicates)
    
    # 生成训练数据
    print(f"🔄 生成训练样本...")
    for _ in tqdm(range(n_samples), desc="生成训练数据", unit="样本"):
        # 随机选择一个y_pred
        y_pred = random.choice(enum_predicates)
        
        # 随机生成一个规则
        current_preds = [y_pred]
        unused = list(feature_predicates)
        
        # 随机添加谓词
        rule_length = random.randint(1, max_depth)
        for _ in range(rule_length - 1):
            if not unused:
                break
            pred = random.choice(unused)
            current_preds.append(pred)
            unused.remove(pred)
        
        # 评估规则
        support, confidence = evaluate_rule(df, current_preds)
        reward = support * confidence
        
        # 创建特征
        state_features = policy_model._create_state_features(current_preds, df_stats)
        
        # 为每个可能的动作创建特征
        for action_pred in unused[:min(10, len(unused))]:  # 限制动作数量
            action_features = policy_model._create_feature_vector(current_preds, unused, df_stats)
            next_state_features = policy_model._create_state_features(current_preds + [action_pred], df_stats)
            
            training_data.append((state_features, action_features, reward, next_state_features))
    
    # 训练模型
    if training_data:
        policy_model.train(training_data)
        
        # 保存模型
        if model_path:
            policy_model.save_model(model_path)
        
        print(f"✅ 策略模型训练完成，使用 {len(training_data)} 个样本")
    else:
        print("⚠️ 没有生成有效的训练数据")
    
    return policy_model

def run_single_experiment(mcts_df, filtered_predicates, enum_predicates, dist_config, 
                         n_iter, c_param, support_threshold, confidence_threshold, 
                         data_dir, experiment_name, use_policy=True, policy_model=None, alpha=1.0):
    """运行单个参数组合的实验"""
    print(f"\n{'='*60}")
    print(f"🔬 实验: {experiment_name}")
    print(f"   参数: n_iter={n_iter}, c_param={c_param}, alpha={alpha}, support_threshold={support_threshold}, confidence_threshold={confidence_threshold}")
    print(f"   策略模型: {'启用' if use_policy else '禁用'}")
    print(f"{'='*60}")
    
    # 临时修改MCTSNode的best_child方法
    original_best_child = MCTSNode.best_child
    
    def custom_best_child(self, c_param=c_param, alpha=alpha):
        import numpy as np
        if not self.children:
            return None
            
        # 使用UCB进行节点选择，alpha参数影响探索项
        choices_weights = []
        for child in self.children:
            # UCB公式：exploitation + alpha * exploration
            exploitation = child.value / (child.visits + 1e-6)
            exploration = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            ucb_score = exploitation + alpha * exploration
            choices_weights.append(ucb_score)
        
        return self.children[np.argmax(choices_weights)]
    
    MCTSNode.best_child = custom_best_child
    
    try:
        # 策略1: 全局规则发现
        print(f"🔍 策略1: 全局规则发现...")
        mcts_results = mcts_rule_discovery_yroot(
            mcts_df, 
            filtered_predicates, 
            enum_predicates, 
            max_depth=6,
            n_iter=n_iter,  # 使用实验参数
            n_workers=dist_config['n_workers'],
            use_parallel=dist_config['use_multiprocessing'],
            use_policy=use_policy,  # 使用策略模型
            policy_model=policy_model,
            c_param=c_param,  # 使用实验参数
            alpha=alpha  # 使用实验参数
        )
        
        # 策略2: 针对特定列的专门规则发现
        print(f"🔍 策略2: 针对特定列的专门规则发现...")
        target_columns = ['genre', 'language', 'format', 'publisher', 'rating', 'availability']
        specialized_results = []
        
        for target_col in target_columns:
            if target_col in mcts_df.columns:
                print(f"  🎯 为{target_col}列发现专门规则...")
                col_vals = mcts_df[target_col].dropna().unique()
                col_predicates = [f'{target_col} = "{val}"' for val in col_vals if pd.notna(val) and str(val).strip() != '']
                
                if col_predicates:
                    col_predicates = col_predicates[:15]
                    print(f"    {target_col}列谓词数: {len(col_predicates)}")
                    
                    col_results = mcts_rule_discovery_yroot(
                        mcts_df,
                        filtered_predicates,
                        col_predicates,
                        max_depth=6,
                        n_iter=n_iter // 2,  # 专门规则使用较少的迭代次数
                        n_workers=dist_config['n_workers'],
                        use_parallel=dist_config['use_multiprocessing'],
                        use_policy=use_policy,  # 使用策略模型
                        policy_model=policy_model,
                        c_param=c_param,  # 使用实验参数
                        alpha=alpha  # 使用实验参数
                    )
                    specialized_results.extend(col_results)
                    print(f"    ✅ {target_col}列发现 {len(col_results)} 个规则")
        
        # 合并所有结果
        all_results = mcts_results + specialized_results
        print(f"✅ 总规则数: {len(all_results)} (全局: {len(mcts_results)}, 专门: {len(specialized_results)})")
        
        # 使用实验参数进行质量过滤
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
            # 保存实验特定的规则文件
            experiment_rules_file = os.path.join(data_dir, f'dcr_mcts_rule_{experiment_name}.csv')
            rules_df.to_csv(experiment_rules_file, index=False)
            print(f'✅ 发现 {len(rules_df)} 个有效规则，保存到: {experiment_rules_file}')
            
            # 规则分布统计
            print(f'📊 规则分布:')
            for col in ['genre', 'language', 'format', 'publisher', 'rating', 'availability']:
                col_rules = rules_df[rules_df['y_pred'].str.contains(col, na=False)]
                print(f'  {col}: {len(col_rules)} 个规则')
        else:
            rules_df = pd.DataFrame(columns=['y_pred', 'best_rule', 'support', 'confidence'])
            experiment_rules_file = os.path.join(data_dir, f'dcr_mcts_rule_{experiment_name}.csv')
            rules_df.to_csv(experiment_rules_file, index=False)
            print('⚠️ 未发现任何有效规则')
        
        # 规则查错和评估
        test_dirty = pd.read_csv(os.path.join(data_dir, 'test_dirty_extend.csv'))
        test_clean = pd.read_csv(os.path.join(data_dir, 'test_extend.csv'))
        
        results = []
        def extract_col_from_predicate(pred):
            import re
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
        experiment_error_file = os.path.join(data_dir, f'dcr_rule_error_detect_{experiment_name}.csv')
        if not error_df.empty:
            error_df.to_csv(experiment_error_file, index=False)
        else:
            error_df = pd.DataFrame(columns=['rule_id', 'y_pred', 'best_rule', 'error_cell', 'error_count'])
            error_df.to_csv(experiment_error_file, index=False)
        
        # 计算评估指标
        pred_cells = set()
        for cells in error_df['error_cell']:
            for cell in json.loads(cells):
                row_idx, col_name = cell
                if row_idx >= len(test_clean) or col_name not in test_clean.columns:
                    continue
                pred_cells.add(tuple(cell))
        
        real_cells = set()
        exclude_cols = [col for col in test_clean.columns if "embed_cluster" in col or "img_category" in col]
        
        print(f"🔍 检测实际错误...")
        for i in tqdm(range(len(test_clean)), desc="检测实际错误", unit="行"):
            for col in test_clean.columns:
                # 跳过不应该有错误的列
                if col in exclude_cols:
                    continue
                clean_val = test_clean.at[i, col]
                dirty_val = test_dirty.at[i, col]
                # 正确处理NaN值比较
                if pd.isna(clean_val) and pd.isna(dirty_val):
                    continue  # 两个都是NaN，不算错误
                elif pd.isna(clean_val) or pd.isna(dirty_val):
                    real_cells.add((i, col))  # 一个NaN一个非NaN，算错误
                elif str(clean_val).strip() != str(dirty_val).strip():
                    real_cells.add((i, col))  # 字符串比较，去除空格
        
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
        
        # 保存实验结果
        experiment_metrics_file = os.path.join(data_dir, f'dcr_rule_error_metrics_{experiment_name}.txt')
        with open(experiment_metrics_file, 'w', encoding='utf-8') as f:
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
        
        print(f'✅ 实验 {experiment_name} 完成:')
        print(f'   Precision: {precision:.4f}')
        print(f'   Recall: {recall:.4f}')
        print(f'   F1: {f1:.4f}')
        print(f'   Accuracy: {accuracy:.4f}')
        
        return {
            'experiment_name': experiment_name,
            'n_iter': n_iter,
            'c_param': c_param,
            'alpha': alpha,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'rules_count': len(rules_df),
            'errors_detected': len(pred_cells),
            'actual_errors': len(real_cells)
        }
        
    finally:
        # 恢复原始的best_child方法
        MCTSNode.best_child = original_best_child

def run_parameter_experiments(mcts_df, filtered_predicates, enum_predicates, dist_config, data_dir, use_policy=True, policy_model=None):
    """运行参数实验"""
    print(f"\n{'='*80}")
    print(f"🔬 开始参数敏感性实验")
    print(f"   策略模型: {'启用' if use_policy else '禁用'}")
    print(f"{'='*80}")
    
    # 实验参数配置
    experiments = []
    
    # 实验1: 不同迭代次数 (𝐼max)
    n_iter_values = [100, 500, 1000, 5000, 10000]
    for n_iter in n_iter_values:
        experiments.append({
            'name': f'iter_{n_iter}',
            'n_iter': n_iter,
            'c_param': 1.4,  # 固定c_param
            'support_threshold': 0.2,
            'confidence_threshold': 0.65
        })
    
    # 实验2: 不同c_param值
    c_param_values = [0.1, 0.5, 1.0, 1.4, 2.0, 3.0]
    for c_param in c_param_values:
        experiments.append({
            'name': f'cparam_{c_param}',
            'n_iter': 10000,  # 固定n_iter
            'c_param': c_param,
            'support_threshold': 0.2,
            'confidence_threshold': 0.65
        })
    
    # 实验3: 不同alpha值 (探索与利用平衡参数)
    alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    for alpha in alpha_values:
        experiments.append({
            'name': f'alpha_{alpha}',
            'n_iter': 10000,  # 固定n_iter
            'c_param': 1.4,   # 固定c_param
            'alpha': alpha,
            'support_threshold': 0.2,
            'confidence_threshold': 0.65
        })
    
    # 运行所有实验
    results = []
    for i, exp_config in enumerate(experiments):
        print(f"\n📊 进度: {i+1}/{len(experiments)} ({((i+1)/len(experiments)*100):.1f}%)")
        
        try:
            result = run_single_experiment(
                mcts_df, filtered_predicates, enum_predicates, dist_config,
                exp_config['n_iter'], exp_config['c_param'], exp_config['support_threshold'], exp_config['confidence_threshold'],
                data_dir, exp_config['name'], use_policy, policy_model, exp_config.get('alpha', 1.0)
            )
            results.append(result)
        except Exception as e:
            print(f"❌ 实验 {exp_config['name']} 失败: {e}")
            results.append({
                'experiment_name': exp_config['name'],
                'n_iter': exp_config['n_iter'],
                'c_param': exp_config['c_param'],
                'alpha': exp_config.get('alpha', 1.0),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'rules_count': 0,
                'errors_detected': 0,
                'actual_errors': 0
            })
    
    # 保存实验结果汇总
    results_df = pd.DataFrame(results)
    results_summary_file = os.path.join(data_dir, 'parameter_experiments_summary.csv')
    results_df.to_csv(results_summary_file, index=False)
    print(f"\n✅ 所有实验完成，结果保存到: {results_summary_file}")
    
    # 打印实验结果分析
    print(f"\n{'='*80}")
    print(f"📊 参数实验结果分析")
    print(f"{'='*80}")
    
    # 分析迭代次数影响
    iter_results = results_df[results_df['experiment_name'].str.startswith('iter_')]
    if not iter_results.empty:
        print(f"\n🔍 迭代次数 (𝐼max) 影响分析:")
        for _, row in iter_results.iterrows():
            print(f"  n_iter={row['experiment_name'].split('_')[1]}: Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, F1={row['f1']:.4f}, Accuracy={row['accuracy']:.4f}")
    
    # 分析c_param影响
    cparam_results = results_df[results_df['experiment_name'].str.startswith('cparam_')]
    if not cparam_results.empty:
        print(f"\n🔍 c_param 影响分析:")
        for _, row in cparam_results.iterrows():
            print(f"  c_param={row['experiment_name'].split('_')[1]}: Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, F1={row['f1']:.4f}, Accuracy={row['accuracy']:.4f}")
    
    # 分析alpha影响
    alpha_results = results_df[results_df['experiment_name'].str.startswith('alpha_')]
    if not alpha_results.empty:
        print(f"\n🔍 alpha 影响分析:")
        for _, row in alpha_results.iterrows():
            print(f"  alpha={row['experiment_name'].split('_')[1]}: Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, F1={row['f1']:.4f}, Accuracy={row['accuracy']:.4f}")
    
    # 找出最佳参数组合
    best_result = results_df.loc[results_df['f1'].idxmax()]
    print(f"\n🏆 最佳参数组合 (基于F1分数):")
    print(f"  实验名称: {best_result['experiment_name']}")
    print(f"  n_iter: {best_result.get('n_iter', 'N/A')}")
    print(f"  c_param: {best_result.get('c_param', 'N/A')}")
    print(f"  alpha: {best_result.get('alpha', 'N/A')}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  规则数量: {best_result['rules_count']}")
    print(f"  检测错误数: {best_result['errors_detected']}")
    print(f"  实际错误数: {best_result['actual_errors']}")
    
    return results_df

def main():
    """主函数"""
    print("🚀 Starting Multimodal DCRLearner Pipeline")
    
    # 获取分布式配置
    dist_config = get_distributed_config()
    print(f"🔧 分布式配置: {dist_config}")
    
    # 使用新的数据路径
    data_dir = "/data_nas/DCR/split_addnoise/goodreads_test"
    
    # 添加实验模式选择
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--experiment':
        print("🔬 启用参数实验模式")
        run_experiment_mode = True
    else:
        print("📊 启用标准模式")
        run_experiment_mode = False
    
    print(f"📁 Using data directory: {data_dir}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # 检查必要的文件是否存在
    required_files = ['train_clean.csv', 'test_clean.csv', 'test_dirty.csv']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return
        else:
            print(f"✅ Found {file}")

    # ========== 新增：数据类型识别与embedding集成 ==========
    # train_csv = os.path.join(data_dir, 'train_clean.csv')
    # test_csv = os.path.join(data_dir, 'test_clean.csv')
    # imgs_dir = os.path.join(data_dir, 'imgs')
    # if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    #     print(f"❌ train_clean.csv 或 test_clean.csv 不存在")
    #     return
    # df_train = pd.read_csv(train_csv)
    # df_test = pd.read_csv(test_csv)
    # df_all = pd.concat([df_train, df_test], ignore_index=True)
    # print(f"📊 合并后数据 shape: {df_all.shape}")

    # # 初始化模型
    # model = MultimodalModels(device='cuda')

    # # 自动识别列类型
    # col_types = {}
    # for col in df_all.columns:
    #     if "img_path" in col:
    #         col_types[col] = "img_path"
    #     elif df_all[col].dtype in ['float64', 'int64']:
    #         col_types[col] = "numeric"
    #     elif df_all[col].dtype == 'object':
    #         unique_count = df_all[col].nunique()
    #         avg_len = df_all[col].astype(str).apply(len).mean()
    #         if unique_count < 20:
    #             col_types[col] = "enum"
    #         elif avg_len > 20:
    #             col_types[col] = "text"
    #         else:
    #             col_types[col] = "enum"
    # print(f"🔍 列类型识别: {col_types}")

    # # 图片 embedding
    # if "img_path" in col_types.values():
    #     img_col = [col for col, typ in col_types.items() if typ == "img_path"][0]
    #     img_paths = df_all[img_col].apply(lambda x: os.path.join(imgs_dir, str(x)))
    #     img_embeds = []
    #     img_categories = []
    #     for path in img_paths:
    #         img_feat = model.extract_image_features(path)
    #         img_category = model.extract_image_category(path)
    #         if img_feat is not None:
    #             img_embeds.append(img_feat.cpu().numpy())
    #         else:
    #             img_embeds.append(None)
    #         img_categories.append(img_category)
    #     df_all["img_embedding"] = img_embeds
    #     df_all["img_category"] = img_categories
    #     print(f"✅ 图片embedding和类别已添加")

    #     # 层次聚类前PCA降维
    #     from sklearn.decomposition import PCA
    #     from scipy.cluster.hierarchy import linkage, fcluster
    #     import numpy as np
    #     valid_indices = [i for i, emb in enumerate(img_embeds) if emb is not None]
    #     valid_embeds = np.array([img_embeds[i].flatten() for i in valid_indices])
    #     if len(valid_embeds) > 1:
    #         pca = PCA(n_components=64)
    #         reduced_embeds = pca.fit_transform(valid_embeds)
    #         Z = linkage(reduced_embeds, method='ward')
    #         max_clusters = min(10, len(valid_embeds))
    #         clusters = fcluster(Z, max_clusters, criterion='maxclust')
    #         img_embed_cluster = [None] * len(img_embeds)
    #         for idx, c in zip(valid_indices, clusters):
    #             img_embed_cluster[idx] = int(c)
    #         df_all[f"img_embed_cluster"] = img_embed_cluster
    #         print(f"✅ 已完成图片embedding的PCA降维+层次聚类，聚类数: {max(clusters)}")
    #     else:
    #         df_all["img_embed_cluster"] = [None] * len(img_embeds)
    #         print(f"⚠️ 有效图片embedding数量不足，未聚类")

    # # 文本 embedding
    # for col, typ in col_types.items():
    #     if typ == "text":
    #         text_embeds = []
    #         for text in df_all[col].astype(str):
    #             text_feat = model.extract_text_features(text)
    #             text_embeds.append(text_feat.cpu().numpy())
    #         df_all[f"text_embedding_{col}"] = text_embeds
    #         print(f"✅ 文本embedding已添加: {col}")

    #         # 文本embedding层次聚类前PCA降维
    #         from sklearn.decomposition import PCA
    #         from scipy.cluster.hierarchy import linkage, fcluster
    #         import numpy as np
    #         valid_indices = [i for i, emb in enumerate(text_embeds) if emb is not None]
    #         valid_embeds = np.array([text_embeds[i].flatten() for i in valid_indices])
    #         if len(valid_embeds) > 1:
    #             pca = PCA(n_components=64)
    #             reduced_embeds = pca.fit_transform(valid_embeds)
    #             Z = linkage(reduced_embeds, method='ward')
    #             max_clusters = min(10, len(valid_embeds))
    #             clusters = fcluster(Z, max_clusters, criterion='maxclust')
    #             text_embed_cluster = [None] * len(text_embeds)
    #             for idx, c in zip(valid_indices, clusters):
    #                 text_embed_cluster[idx] = int(c)
    #             df_all[f"text_{col}_embed_cluster"] = text_embed_cluster
    #             print(f"✅ 已完成文本embedding的PCA降维+层次聚类: {col}，聚类数: {max(clusters)}")
    #         else:
    #             df_all[f"text_{col}_embed_cluster"] = [None] * len(text_embeds)
    #             print(f"⚠️ 有效文本embedding数量不足，未聚类: {col}")

    # # 保存为pkl
    # out_pkl = os.path.join(data_dir, "train_with_embeddings.pkl")
    # df_all.to_pickle(out_pkl)
    # print(f"✅ 已保存带embedding的数据: {out_pkl}")

    # # 保存为csv，分别拆分train/test
    # n_train = len(df_train)
    # df_train_extend = df_all.iloc[:n_train].copy()
    # df_test_extend = df_all.iloc[n_train:].copy()
    # # 只保存可序列化的列（去除高维embedding列）
    # drop_cols = [col for col in df_all.columns if isinstance(df_all[col].iloc[0], (np.ndarray, list, dict, torch.Tensor))]
    # df_train_csv = df_train_extend.drop(columns=drop_cols)
    # df_test_csv = df_test_extend.drop(columns=drop_cols)
    # df_train_csv.to_csv(os.path.join(data_dir, "train_extend.csv"), index=False)
    # df_test_csv.to_csv(os.path.join(data_dir, "test_extend.csv"), index=False)
    # print(f"✅ 已保存扩展特征的csv: train_extend.csv, test_extend.csv")

    
    df_train_csv=pd.read_csv(os.path.join(data_dir, "train_extend.csv"))
    df_test_csv=pd.read_csv(os.path.join(data_dir, "test_extend.csv"))
    test_dirty=pd.read_csv(os.path.join(data_dir, "test_dirty.csv"))
    
    # test_dirty聚类标签直接用test_clean的（假设一一对应）
    for col in df_test_csv.columns:
        if ("embed_cluster" in col or "img_category" in col) and col not in test_dirty.columns:
            # 确保数据类型一致，避免浮点数vs整数的比较问题
            test_dirty[col] = df_test_csv[col].astype(str).values
    test_dirty.to_csv(os.path.join(data_dir, "test_dirty_extend.csv"), index=False)
    print(f"✅ 已保存扩展特征的csv: test_dirty_extend.csv")


    # 后续pipeline用train_extend.csv、test_dirty_extend.csv
    out_csv = os.path.join(data_dir, "train_extend.csv")

    # 构造谓词并保存
    pc = PredicateConstructor(out_csv)
    predicates = pc.construct_predicates()
    with open(os.path.join(data_dir, "predicates.txt"), "w", encoding="utf-8") as f:
        for p in predicates:
            f.write(p + "\n")
    print(f"✅ 已保存所有构造谓词到: {os.path.join(data_dir, 'predicates.txt')}")

    # MCTS规则发现（以枚举型谓词为Y predicate）
    mcts_df = pd.read_csv(out_csv)
    with open(os.path.join(data_dir, 'predicates.txt'), 'r', encoding='utf-8') as f:
        mcts_predicates = [line.strip() for line in f if line.strip()]
    # 支持度筛选，减少MCTS搜索空间
    support_filter_threshold = 0.05  # 降低阈值，只保留支持度大于0.5%的谓词
    max_predicates = 1000  # 限制最大谓词数量
    filtered_predicates = []
    for p in mcts_predicates:
        mask = predicate_mask(mcts_df, p)
        support = mask.sum() / len(mcts_df) if len(mcts_df) > 0 else 0
        if support >= support_filter_threshold:
            filtered_predicates.append(p)
        if len(filtered_predicates) >= max_predicates:  # 达到上限就停止
            break
    print(f'✅ 支持度筛选后谓词数: {len(filtered_predicates)} (原始: {len(mcts_predicates)})')
    # 枚举型谓词筛选（如 t0.col = ... where col为枚举型）
    enum_predicates = [p for p in filtered_predicates if re.search(r'=\s*"', p)]
    print(f'✅ 枚举型谓词数: {len(enum_predicates)}')
    
    # 为Goodreads数据集构建专门的谓词
    
    # 为genre列添加专门的谓词（图书类型）
    if 'genre' in mcts_df.columns:
        genre_vals = mcts_df['genre'].dropna().unique()
        # 只取前15个最常见的类型，避免谓词爆炸
        genre_vals = genre_vals[:15] if len(genre_vals) > 15 else genre_vals
        for val in genre_vals:
            if pd.notna(val) and str(val).strip() != '':
                enum_predicates.append(f'genre = "{val}"')
        print(f'✅ 为genre列添加了 {len(genre_vals)} 个谓词')
        
        # 添加genre的错误模式谓词（只添加最重要的几个）
        important_genres = [
            'genre = "Fiction"',      # 检测小说相关的错误
            'genre = "Non-Fiction"',  # 检测非小说相关的错误
            'genre = "Mystery"',      # 检测悬疑相关的错误
            'genre = "Romance"',      # 检测言情相关的错误
            'genre = "Science Fiction"',  # 检测科幻相关的错误
        ]
        for pred in important_genres:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了genre重要错误模式谓词')
    
    # 为language列添加专门的谓词（语言）
    if 'language' in mcts_df.columns:
        language_vals = mcts_df['language'].dropna().unique()
        for val in language_vals:
            if pd.notna(val) and str(val).strip() != '':
                enum_predicates.append(f'language = "{val}"')
        print(f'✅ 为language列添加了 {len(language_vals)} 个谓词')
        
        # 添加language的错误模式谓词
        important_languages = [
            'language = "English"',   # 检测英语相关的错误
            'language = "Spanish"',   # 检测西班牙语相关的错误
            'language = "French"',    # 检测法语相关的错误
            'language = "German"',    # 检测德语相关的错误
        ]
        for pred in important_languages:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了language重要错误模式谓词')
    
    # 为format列添加专门的谓词（图书格式）
    if 'format' in mcts_df.columns:
        format_vals = mcts_df['format'].dropna().unique()
        for val in format_vals:
            if pd.notna(val) and str(val).strip() != '':
                enum_predicates.append(f'format = "{val}"')
        print(f'✅ 为format列添加了 {len(format_vals)} 个谓词')
        
        # 添加format的错误模式谓词
        important_formats = [
            'format = "Paperback"',   # 检测平装本相关的错误
            'format = "Hardcover"',   # 检测精装本相关的错误
            'format = "Ebook"',       # 检测电子书相关的错误
            'format = "Audiobook"',   # 检测有声书相关的错误
        ]
        for pred in important_formats:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了format重要错误模式谓词')
    
    # 为publisher列添加专门的谓词（出版社）
    if 'publisher' in mcts_df.columns:
        publisher_vals = mcts_df['publisher'].dropna().unique()
        # 只取前10个最常见的出版社，避免谓词爆炸
        publisher_vals = publisher_vals[:10] if len(publisher_vals) > 10 else publisher_vals
        for val in publisher_vals:
            if pd.notna(val) and str(val).strip() != '':
                enum_predicates.append(f'publisher = "{val}"')
        print(f'✅ 为publisher列添加了 {len(publisher_vals)} 个谓词')
        
        # 添加publisher的错误模式谓词
        important_publishers = [
            'publisher = "Penguin"',      # 检测企鹅出版社相关的错误
            'publisher = "Random House"', # 检测兰登书屋相关的错误
            'publisher = "HarperCollins"', # 检测哈珀柯林斯相关的错误
        ]
        for pred in important_publishers:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了publisher重要错误模式谓词')
    
    # 为rating列添加专门的谓词（评分）
    if 'rating' in mcts_df.columns:
        rating_vals = mcts_df['rating'].dropna().unique()
        for val in rating_vals:
            if pd.notna(val):
                enum_predicates.append(f'rating = "{val}"')
        print(f'✅ 为rating列添加了 {len(rating_vals)} 个谓词')
        
        # 添加rating的错误模式谓词
        important_ratings = [
            'rating = "5"',  # 检测5星评分相关的错误
            'rating = "4"',  # 检测4星评分相关的错误
            'rating = "3"',  # 检测3星评分相关的错误
        ]
        for pred in important_ratings:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了rating重要错误模式谓词')
    
    # 为availability列添加专门的谓词（可用性）
    if 'availability' in mcts_df.columns:
        availability_vals = mcts_df['availability'].dropna().unique()
        for val in availability_vals:
            if pd.notna(val):
                enum_predicates.append(f'availability = "{val}"')
        print(f'✅ 为availability列添加了 {len(availability_vals)} 个谓词')
        
        # 添加availability的错误模式谓词
        availability_errors = [
            'availability = "Available"',   # 检测可用相关的错误
            'availability = "Out of Stock"', # 检测缺货相关的错误
        ]
        for pred in availability_errors:
            if pred not in enum_predicates:
                enum_predicates.append(pred)
        print(f'✅ 添加了availability错误模式谓词')
    
    print(f'✅ 最终枚举型谓词数: {len(enum_predicates)}')
    # 限制枚举型谓词数量，避免MCTS搜索过慢
    # if len(enum_predicates) > 50:
    #     enum_predicates = enum_predicates[:50]
    #     print(f'✅ 限制枚举型谓词数为: {len(enum_predicates)}')
    # 使用分布式MCTS规则发现
    print(f"🎯 开始分布式MCTS规则发现...")
    print(f"   数据规模: {len(mcts_df)}行, {len(filtered_predicates)}个谓词, {len(enum_predicates)}个y_pred")
    
    # 训练或加载策略模型
    policy_model = None
    model_path = os.path.join(data_dir, 'mcts_policy_model.pkl')
    
    # 用户可以选择是否使用策略模型
    # 如果遇到卡住问题，可以设置为False快速切换到传统MCTS
    use_policy_model = True  # 可以通过环境变量或配置文件控制
    
    # 快速修复：如果遇到分布式策略模型问题，可以设置为False
    if os.environ.get('DISABLE_POLICY_MODEL', 'false').lower() == 'true':
        use_policy_model = False
        print(f"⚠️ 检测到环境变量DISABLE_POLICY_MODEL=true，禁用策略模型")
    
    if use_policy_model:
        if os.path.exists(model_path):
            print(f"📥 加载已有策略模型: {model_path}")
            policy_model = ValuePolicyModel()
            policy_model.load_model(model_path)
        else:
            print(f"🎯 训练新的策略模型...")
            policy_model = train_policy_model(
                mcts_df, 
                filtered_predicates, 
                enum_predicates, 
                max_depth=6, 
                n_samples=2000,  # 增加训练样本数量
                model_path=model_path
            )
    else:
        print(f"⚠️ 跳过策略模型训练，使用传统MCTS")
    
    # 策略1: 全局规则发现（使用策略模型）
    print(f"🔍 策略1: 全局规则发现（使用策略模型）...")
    
    # 统一使用多线程模式，简化策略模型
    print(f"🎯 使用多线程MCTS模式，简化策略模型")
    use_parallel_for_policy = dist_config['use_multiprocessing']
    n_workers_for_policy = dist_config['n_workers']
    
    # 简化策略模型使用：只使用启发式策略，不使用机器学习模型
    if use_policy_model and policy_model and policy_model.is_trained:
        use_policy_flag = True
        print(f"✅ 使用启发式策略模型")
    else:
        use_policy_flag = False
        policy_model = None
        print(f"⚠️ 使用传统随机策略")
    
    mcts_results = mcts_rule_discovery_yroot(
        mcts_df, 
        filtered_predicates, 
        enum_predicates, 
        max_depth=6,  # 进一步增加搜索深度以发现更复杂的规则
        n_iter=10000,  # 大幅增加迭代次数以提高召回率
        n_workers=n_workers_for_policy,
        use_parallel=use_parallel_for_policy,
        use_policy=use_policy_flag,  # 根据配置决定是否启用策略模型
        policy_model=policy_model,
        c_param=1.4,  # MCTS探索参数
        alpha=1.0  # 探索与利用平衡参数
    )
    
    # 策略2: 针对特定列的专门规则发现
    print(f"🔍 策略2: 针对特定列的专门规则发现...")
    target_columns = ['genre', 'language', 'format', 'publisher', 'rating', 'availability']
    specialized_results = []
    
    for target_col in tqdm(target_columns, desc="专门规则发现", unit="列"):
        if target_col in mcts_df.columns:
            print(f"  🎯 为{target_col}列发现专门规则...")
            # 为该列创建专门的y_pred
            col_vals = mcts_df[target_col].dropna().unique()
            col_predicates = [f'{target_col} = "{val}"' for val in col_vals if pd.notna(val) and str(val).strip() != '']
            
            if col_predicates:
                # 限制谓词数量以避免搜索过慢
                col_predicates = col_predicates[:15]  # 增加谓词数量以捕获更多模式
                print(f"    {target_col}列谓词数: {len(col_predicates)}")
                
                # 为该列进行专门的MCTS搜索（使用策略模型）
                col_results = mcts_rule_discovery_yroot(
                    mcts_df,
                    filtered_predicates,
                    col_predicates,
                    max_depth=6,  # 进一步增加深度以发现更复杂的规则
                    n_iter=5000,  # 增加迭代次数以提高发现概率
                    n_workers=n_workers_for_policy,  # 使用与全局规则发现相同的设置
                    use_parallel=use_parallel_for_policy,  # 使用与全局规则发现相同的设置
                    use_policy=use_policy_flag,  # 根据配置决定是否启用策略模型
                    policy_model=policy_model,
                    c_param=1.4,  # MCTS探索参数
                    alpha=1.0  # 探索与利用平衡参数
                )
                specialized_results.extend(col_results)
                print(f"    ✅ {target_col}列发现 {len(col_results)} 个规则")
    
    # 合并所有结果
    all_results = mcts_results + specialized_results
    print(f"✅ 总规则数: {len(all_results)} (全局: {len(mcts_results)}, 专门: {len(specialized_results)})")
    # 阈值设置 - 适度优化以提高召回率
    support_threshold = 0.2  # 适度降低支持度阈值以发现更多规则
    confidence_threshold = 0.65  # 适度降低置信度阈值以提高召回率
    # 保存为结构化csv
    rules_data = []
    for y_pred, rule, support, confidence in all_results:
        if support >= support_threshold and confidence >= confidence_threshold:
            # 额外的质量过滤：避免过于宽泛的规则
            rule_complexity = len(rule) - 1  # 前提条件的数量
            if rule_complexity >= 1:  # 至少需要1个前提条件（放宽要求）
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
        for col in ['genre', 'language', 'format', 'publisher', 'rating', 'availability']:
            col_rules = rules_df[rules_df['y_pred'].str.contains(col, na=False)]
            print(f'  {col}: {len(col_rules)} 个规则')
    else:
        rules_df = pd.DataFrame(columns=['y_pred', 'best_rule', 'support', 'confidence'])
        rules_df.to_csv(os.path.join(data_dir, 'dcr_mcts_rule.csv'), index=False)
        print('⚠️ 未发现任何有效规则')
    print(f'✅ 已保存结构化规则表到: {os.path.join(data_dir, "dcr_mcts_rule.csv")}')

    # 规则查错：输出error_cell (行号, 列名)
    rules_df = pd.read_csv(os.path.join(data_dir, 'dcr_mcts_rule.csv'))
    test_dirty = pd.read_csv(os.path.join(data_dir, 'test_dirty_extend.csv'))
    test_clean = pd.read_csv(os.path.join(data_dir, 'test_extend.csv'))

    results = []
    def extract_col_from_predicate(pred):
        import re
        m = re.match(r'(\w+)\s*[=!<>]+\s*.+', pred)
        if m:
            return m.group(1)
        return None
    # 排除不应该有错误的列
    exclude_cols = [col for col in test_dirty.columns if "embed_cluster" in col or "img_category" in col]
    
    for idx, row in tqdm(rules_df.iterrows(), total=len(rules_df), desc="规则查错", unit="规则"):
        y_pred = row['y_pred']
        best_rule = row['best_rule']
        # 解析前提谓词
        premise_preds = [p.strip() for p in best_rule.split('^') if p.strip()]
        # 结论谓词
        conclusion_pred = y_pred
        conclusion_col = extract_col_from_predicate(conclusion_pred)
        if conclusion_col is None or conclusion_col in exclude_cols:
            continue  # 跳过无法提取列名的规则或不应该有错误的列
        # 前提mask
        mask = np.ones(len(test_dirty), dtype=bool)
        for pred in premise_preds:
            mask = mask & predicate_mask(test_dirty, pred)
        # 结论mask
        mask_conclusion = predicate_mask(test_dirty, conclusion_pred)
        # 查错：前提成立但结论不成立的样本
        error_mask = mask & (~mask_conclusion)
        
        # 过滤掉NaN值和误报
        for i in range(len(test_dirty)):
            if error_mask[i]:
                clean_val = test_clean.iloc[i][conclusion_col] if i < len(test_clean) else None
                dirty_val = test_dirty.iloc[i][conclusion_col]
                # 如果两个值都是NaN，不算错误
                if pd.isna(clean_val) and pd.isna(dirty_val):
                    error_mask[i] = False
                # 如果两个值相同，不算错误（避免误报）
                elif not pd.isna(clean_val) and not pd.isna(dirty_val):
                    if str(clean_val).strip() == str(dirty_val).strip():
                        error_mask[i] = False
        
        # 使用位置索引而不是DataFrame索引，确保一致性
        error_positions = [i for i, is_error in enumerate(error_mask) if is_error]
        # 输出error_cell
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
        # 明确指定列名，写入表头
        error_df = pd.DataFrame(columns=['rule_id', 'y_pred', 'best_rule', 'error_cell', 'error_count'])
        error_df.to_csv(os.path.join(data_dir, 'dcr_rule_error_detect.csv'), index=False)
    print('✅ 已保存规则查错结果到 dcr_rule_error_detect.csv')

    # 规则查错评估：与test_clean.csv对比，计算F1/recall/precision/accuracy
    import ast
    error_file = os.path.join(data_dir, 'dcr_rule_error_detect.csv')
    # 判断文件是否为空或无列
    if os.path.getsize(error_file) == 0 or pd.read_csv(error_file).shape[1] == 0:
        print("⚠️ 查错结果文件为空或无列，跳过评估。")
        with open(os.path.join(data_dir, 'dcr_rule_error_metrics.txt'), 'w', encoding='utf-8') as f:
            f.write('Precision: 0.0000\nRecall: 0.0000\nF1: 0.0000\nAccuracy: 0.0000\n')
        return

    test_clean = pd.read_csv(os.path.join(data_dir, 'test_extend.csv'))    
    test_dirty = pd.read_csv(os.path.join(data_dir, 'test_dirty_extend.csv'))
    
    # 打印数据框大小信息
    print(f"📊 数据框大小信息:")
    print(f"   test_clean.shape: {test_clean.shape}")
    print(f"   test_dirty.shape: {test_dirty.shape}")
    print(f"   test_clean.columns: {list(test_clean.columns)}")
    print(f"   test_dirty.columns: {list(test_dirty.columns)}") 
    
    # 预测为正的cell集合
    pred_cells = set()
    for cells in error_df['error_cell']:
        for cell in json.loads(cells):
            row_idx, col_name = cell
            # 验证索引在有效范围内
            if row_idx >= len(test_clean):
                print(f"⚠️ 警告：预测错误索引超出范围: 行{row_idx}, 列{col_name} (test_clean.shape={test_clean.shape})")
                continue
            if col_name not in test_clean.columns:
                print(f"⚠️ 警告：预测错误列不存在: 行{row_idx}, 列{col_name}")
                continue
            pred_cells.add(tuple(cell))
    
    print(f"📊 有效预测错误数: {len(pred_cells)}")
    
    # 实际为正的cell集合
    real_cells = set()
    # 排除不应该有错误的列（embedding聚类列和图片类别列）
    exclude_cols = [col for col in test_clean.columns if "embed_cluster" in col or "img_category" in col]
    print(f"🔍 排除的列（不应该有错误）: {exclude_cols}")
    
    print(f"🔍 检测实际错误...")
    for i in tqdm(range(len(test_clean)), desc="检测实际错误", unit="行"):
        for col in test_clean.columns:
            # 跳过不应该有错误的列
            if col in exclude_cols:
                continue
            clean_val = test_clean.at[i, col]
            dirty_val = test_dirty.at[i, col]
            # 正确处理NaN值比较
            if pd.isna(clean_val) and pd.isna(dirty_val):
                continue  # 两个都是NaN，不算错误
            elif pd.isna(clean_val) or pd.isna(dirty_val):
                real_cells.add((i, col))  # 一个NaN一个非NaN，算错误
            elif str(clean_val).strip() != str(dirty_val).strip():
                real_cells.add((i, col))  # 字符串比较，去除空格
    
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
    
    # ========== 错误统计分析 ==========
    print(f"\n📊 错误统计分析:")
    print(f"总单元格数: {total_cells}")
    print(f"实际错误数: {len(real_cells)}")
    print(f"预测错误数: {len(pred_cells)}")
    print(f"真阳性(TP): {TP}")
    print(f"假阳性(FP): {FP}")
    print(f"假阴性(FN): {FN}")
    print(f"真阴性(TN): {TN}")
    
    # 分析查到的错误（TP）
    if TP > 0:
        print(f"\n✅ 查到的错误 (TP={TP}):")
        tp_cells = list(pred_cells & real_cells)
        tp_cells.sort()
        for i, (row_idx, col_name) in enumerate(tp_cells[:10]):  # 只显示前10个
            # 检查索引是否在有效范围内
            if row_idx < len(test_clean) and col_name in test_clean.columns:
                clean_val = test_clean.at[row_idx, col_name]
                dirty_val = test_dirty.at[row_idx, col_name]
                # 正确处理NaN值显示
                clean_str = str(clean_val) if not pd.isna(clean_val) else 'NaN'
                dirty_str = str(dirty_val) if not pd.isna(dirty_val) else 'NaN'
                print(f"  {i+1}. 行{row_idx}, 列{col_name}: '{clean_str}' -> '{dirty_str}'")
            else:
                print(f"  {i+1}. 行{row_idx}, 列{col_name}: 索引超出范围 (test_clean.shape={test_clean.shape})")
        if len(tp_cells) > 10:
            print(f"  ... 还有 {len(tp_cells)-10} 个查到的错误")
    
    # 分析未查到的错误（FN）
    if FN > 0:
        print(f"\n❌ 未查到的错误 (FN={FN}):")
        fn_cells = list(real_cells - pred_cells)
        fn_cells.sort()
        for i, (row_idx, col_name) in enumerate(fn_cells[:10]):  # 只显示前10个
            # 检查索引是否在有效范围内
            if row_idx < len(test_clean) and col_name in test_clean.columns:
                clean_val = test_clean.at[row_idx, col_name]
                dirty_val = test_dirty.at[row_idx, col_name]
                # 正确处理NaN值显示
                clean_str = str(clean_val) if not pd.isna(clean_val) else 'NaN'
                dirty_str = str(dirty_val) if not pd.isna(dirty_val) else 'NaN'
                print(f"  {i+1}. 行{row_idx}, 列{col_name}: '{clean_str}' -> '{dirty_str}'")
            else:
                print(f"  {i+1}. 行{row_idx}, 列{col_name}: 索引超出范围 (test_clean.shape={test_clean.shape})")
        if len(fn_cells) > 10:
            print(f"  ... 还有 {len(fn_cells)-10} 个未查到的错误")
    
    # 分析误报的错误（FP）
    if FP > 0:
        print(f"\n⚠️ 误报的错误 (FP={FP}):")
        fp_cells = list(pred_cells - real_cells)
        fp_cells.sort()
        
        # 先显示所有规则，帮助理解误报原因
        print(f"🔍 当前所有规则:")
        print(f"error_df columns: {error_df.columns.tolist()}")
        print(f"rules_df columns: {rules_df.columns.tolist()}")
        for idx, row in error_df.iterrows():
            rule_id = row['rule_id']
            rule_info = f"  Rule_{rule_id}: {row['y_pred']} <- {row['best_rule']}"
            # 从rules_df查support/confidence
            if rule_id < len(rules_df):
                rule_row = rules_df.iloc[rule_id]
                rule_info += f" (support={rule_row['support']:.3f}, confidence={rule_row['confidence']:.3f})"
            print(rule_info)

        # 分析误报的规则来源
        print(f"\n🔍 误报分析 - 检查哪些规则导致了误报:")
        fp_rule_counts = {}
        
        # 为每个误报cell找到对应的规则
        for fp_cell in fp_cells:
            cell_found = False
            for idx, row in error_df.iterrows():
                rule_cells = json.loads(row['error_cell'])
                if fp_cell in rule_cells:
                    rule_info = f"Rule_{row['rule_id']}: {row['y_pred']} <- {row['best_rule']}"
                    fp_rule_counts[rule_info] = fp_rule_counts.get(rule_info, 0) + 1
                    cell_found = True
                    break  # 找到第一个匹配的规则就停止
        
        # 显示导致误报最多的规则
        if fp_rule_counts:
            sorted_rules = sorted(fp_rule_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"📋 导致误报最多的规则:")
            for rule_info, count in sorted_rules[:5]:
                print(f"  {rule_info}: {count}个误报")
            
            # 显示误报分布统计
            print(f"\n📊 误报分布统计:")
            total_fp = len(fp_cells)
            covered_fp = sum(fp_rule_counts.values())
            print(f"  总误报数: {total_fp}")
            print(f"  被规则覆盖的误报数: {covered_fp}")
            print(f"  未找到来源的误报数: {total_fp - covered_fp}")
        else:
            print("  ⚠️ 无法确定误报来源")
            
        # 显示每个误报cell对应的规则
        print(f"\n🔍 误报cell与规则对应关系:")
        for i, fp_cell in enumerate(fp_cells[:10]):  # 只显示前10个
            cell_found = False
            for idx, row in error_df.iterrows():
                rule_cells = json.loads(row['error_cell'])
                if fp_cell in rule_cells:
                    rule_info = f"Rule_{row['rule_id']}: {row['y_pred']}"
                    print(f"  {i+1}. 行{fp_cell[0]}, 列{fp_cell[1]} -> {rule_info}")
                    cell_found = True
                    break
            if not cell_found:
                print(f"  {i+1}. 行{fp_cell[0]}, 列{fp_cell[1]} -> 未找到对应规则")
        if len(fp_cells) > 10:
            print(f"  ... 还有 {len(fp_cells)-10} 个误报cell")
        
        # 显示误报详情
        print(f"\n📋 误报详情:")
        for i, (row_idx, col_name) in enumerate(fp_cells[:10]):  # 只显示前10个
            # 检查索引是否在有效范围内
            if row_idx < len(test_clean) and col_name in test_clean.columns:
                clean_val = test_clean.at[row_idx, col_name]
                dirty_val = test_dirty.at[row_idx, col_name]
                # 正确处理NaN值显示
                clean_str = str(clean_val) if not pd.isna(clean_val) else 'NaN'
                dirty_str = str(dirty_val) if not pd.isna(dirty_val) else 'NaN'
                print(f"  {i+1}. 行{row_idx}, 列{col_name}: '{clean_str}' == '{dirty_str}' (实际无错误)")
                
                # 分析误报原因：检查该行是否满足任何规则的前提条件
                print(f"     🔍 误报原因分析:")
                for idx, row in error_df.iterrows():
                    rule_id = row['rule_id']
                    y_pred = row['y_pred']
                    best_rule = row['best_rule']
                    
                    # 检查这个误报是否由这个规则引起
                    rule_cells = json.loads(row['error_cell'])
                    if (row_idx, col_name) in rule_cells:
                        print(f"       - 由Rule_{rule_id}引起: {y_pred} <- {best_rule}")
                        
                        # 分析规则前提条件
                        if best_rule:
                            premises = best_rule.split(' ^ ')
                            print(f"         前提条件:")
                            for premise in premises:
                                premise = premise.strip()
                                if premise:
                                    # 检查该行是否满足这个前提条件
                                    try:
                                        mask = predicate_mask(test_clean, premise)
                                        if mask.iloc[row_idx]:
                                            print(f"           ✓ {premise} (满足)")
                                        else:
                                            print(f"           ✗ {premise} (不满足)")
                                    except:
                                        print(f"           ? {premise} (无法评估)")
                        break
            else:
                print(f"  {i+1}. 行{row_idx}, 列{col_name}: 索引超出范围 (test_clean.shape={test_clean.shape})")
        if len(fp_cells) > 10:
            print(f"  ... 还有 {len(fp_cells)-10} 个误报")
    
    # 按列统计错误分布
    print(f"\n📈 按列统计错误分布:")
    col_error_stats = {}
    for row_idx, col_name in real_cells:
        if col_name not in col_error_stats:
            col_error_stats[col_name] = {'total': 0, 'detected': 0, 'missed': 0}
        col_error_stats[col_name]['total'] += 1
        if (row_idx, col_name) in pred_cells:
            col_error_stats[col_name]['detected'] += 1
        else:
            col_error_stats[col_name]['missed'] += 1
    
    # 按检测率排序
    sorted_cols = sorted(col_error_stats.items(), 
                        key=lambda x: x[1]['detected']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                        reverse=True)
    
    for col_name, stats in sorted_cols[:10]:  # 显示前10列
        detection_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {col_name}: {stats['detected']}/{stats['total']} ({detection_rate:.2%})")
    
    # 规则质量评估
    print(f"\n🔍 规则质量评估:")
    for idx, row in error_df.iterrows():
        rule_id = row['rule_id']
        y_pred = row['y_pred']
        best_rule = row['best_rule']
        error_cells = json.loads(row['error_cell'])
        
        # 将error_cells转换为元组集合，使其可哈希
        error_cells_set = set(tuple(cell) for cell in error_cells)
        
        # 计算该规则的精确率
        rule_tp = len(error_cells_set & real_cells)
        rule_fp = len(error_cells_set - real_cells)
        rule_precision = rule_tp / (rule_tp + rule_fp) if (rule_tp + rule_fp) > 0 else 0
        
        # 计算该规则的召回率
        rule_fn = len(real_cells - error_cells_set)
        rule_recall = rule_tp / (rule_tp + rule_fn) if (rule_tp + rule_fn) > 0 else 0
        
        print(f"  Rule_{rule_id}: {y_pred} <- {best_rule}")
        print(f"    精确率: {rule_precision:.3f} (TP={rule_tp}, FP={rule_fp})")
        print(f"    召回率: {rule_recall:.3f} (TP={rule_tp}, FN={rule_fn})")
        print(f"    覆盖单元格数: {len(error_cells)}")
        print()
    
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
    
    # 如果启用实验模式，运行参数实验
    if run_experiment_mode:
        print(f"\n{'='*80}")
        print(f"🔬 开始参数敏感性实验")
        print(f"{'='*80}")
        
        # 运行参数实验
        experiment_results = run_parameter_experiments(
            mcts_df, filtered_predicates, enum_predicates, dist_config, data_dir, use_policy_flag, policy_model
        )
        
        print(f"\n✅ 参数实验完成！")
        print(f"📊 实验结果已保存到: {os.path.join(data_dir, 'parameter_experiments_summary.csv')}")
        print(f"🔍 详细分析请查看控制台输出和各个实验文件")


if __name__ == '__main__':
    main() 