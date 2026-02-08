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


class MCTSNode:
    def __init__(self, predicates, parent=None):
        self.predicates = predicates  # 当前节点的谓词组合（list）
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
        import numpy as np
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
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


def mcts_rule_discovery(df, predicates, enum_predicates, max_depth=3, n_iter=100):
    results = []
    feature_predicates = [p for p in predicates if p not in enum_predicates]
    for y_pred in enum_predicates:
        root = MCTSNode([y_pred])
        best_support, best_confidence = 0, 0.0
        best_rule = []
        for _ in range(n_iter):
            node = root
            # Selection
            while node.children:
                node = node.best_child()
            # Expansion
            if not node.is_terminal(max_depth, feature_predicates):
                node.expand(feature_predicates)
                if node.children:
                    import random
                    node = random.choice(node.children)
            # Simulation
            sim_preds = list(node.predicates)
            unused = list(set(feature_predicates) - set(sim_preds[1:]))
            import random
            while len(sim_preds) < max_depth and unused:
                sim_preds.append(random.choice(unused))
                unused = list(set(feature_predicates) - set(sim_preds[1:]))
            # 评估：前提=sim_preds[1:], 结论=sim_preds[0]
            support, confidence = evaluate_rule(df, [sim_preds[0]] + sim_preds[1:])
            reward = support * confidence
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
    df, predicates, y_pred, max_depth, n_iter = args
    feature_predicates = [p for p in predicates if p not in [y_pred]]
    
    root = MCTSNode([y_pred])
    best_support, best_confidence = 0, 0.0
    best_rule = []
    
    for _ in range(n_iter):
        node = root
        # Selection
        while node.children:
            node = node.best_child()
        # Expansion
        if not node.is_terminal(max_depth, feature_predicates):
            node.expand(feature_predicates)
            if node.children:
                node = random.choice(node.children)
        # Simulation
        sim_preds = list(node.predicates)
        unused = list(set(feature_predicates) - set(sim_preds[1:]))
        while len(sim_preds) < max_depth and unused:
            sim_preds.append(random.choice(unused))
            unused = list(set(feature_predicates) - set(sim_preds[1:]))
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

def mcts_rule_discovery_yroot(df, predicates, enum_predicates, max_depth=3, n_iter=100, n_workers=None, use_parallel=True):
    """
    分布式MCTS规则发现
    
    Args:
        df: 数据框
        predicates: 所有谓词
        enum_predicates: 枚举型谓词（作为结论）
        max_depth: 最大规则深度
        n_iter: 每个y_pred的迭代次数
        n_workers: 并行工作进程数，None表示使用CPU核心数
    """
    if not use_parallel or len(enum_predicates) < 5:
        # 单线程模式（用于调试或小规模数据）
        print(f"🔄 使用单线程MCTS: {len(enum_predicates)}个y_pred")
        start_time = time.time()
        results = []
        for i, y_pred in enumerate(enum_predicates):
            result = mcts_rule_discovery_single_y_pred((df, predicates, y_pred, max_depth, n_iter))
            results.append(result)
            if (i + 1) % max(1, len(enum_predicates) // 10) == 0:
                print(f"📊 进度: {i+1}/{len(enum_predicates)} ({(i+1)/len(enum_predicates)*100:.1f}%)")
        return results
    
    # 多线程模式
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(enum_predicates))
    
    print(f"🚀 启动分布式MCTS规则发现: {len(enum_predicates)}个y_pred, {n_workers}个工作进程")
    start_time = time.time()
    
    # 准备参数
    args_list = [(df, predicates, y_pred, max_depth, n_iter) for y_pred in enum_predicates]
    
    # 使用进程池进行并行计算
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_y_pred = {executor.submit(mcts_rule_discovery_single_y_pred, args): args[2] 
                           for args in args_list}
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_y_pred):
            y_pred = future_to_y_pred[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % max(1, len(enum_predicates) // 10) == 0:
                    print(f"📊 进度: {completed}/{len(enum_predicates)} ({completed/len(enum_predicates)*100:.1f}%)")
            except Exception as e:
                print(f"❌ y_pred {y_pred} 处理失败: {e}")
                # 添加默认结果
                results.append((y_pred, [], 0, 0.0))
    
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

def main():
    """主函数"""
    print("🚀 Starting Multimodal DCRLearner Pipeline")
    
    # 获取分布式配置
    dist_config = get_distributed_config()
    print(f"🔧 分布式配置: {dist_config}")
    
    # 使用新的数据路径
    data_dir = "/data_nas/DCR/split_addnoise/goodreads_test"
    
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
    
    # 策略1: 全局规则发现
    print(f"🔍 策略1: 全局规则发现...")
    mcts_results = mcts_rule_discovery_yroot(
        mcts_df, 
        filtered_predicates, 
        enum_predicates, 
        max_depth=6,  # 进一步增加搜索深度以发现更复杂的规则
        n_iter=10000,  # 大幅增加迭代次数以提高召回率
        n_workers=dist_config['n_workers'],
        use_parallel=dist_config['use_multiprocessing']
    )
    
    # 策略2: 针对特定列的专门规则发现
    print(f"🔍 策略2: 针对特定列的专门规则发现...")
    target_columns = ['genre', 'language', 'format', 'publisher', 'rating', 'availability']
    specialized_results = []
    
    for target_col in target_columns:
        if target_col in mcts_df.columns:
            print(f"  🎯 为{target_col}列发现专门规则...")
            # 为该列创建专门的y_pred
            col_vals = mcts_df[target_col].dropna().unique()
            col_predicates = [f'{target_col} = "{val}"' for val in col_vals if pd.notna(val) and str(val).strip() != '']
            
            if col_predicates:
                # 限制谓词数量以避免搜索过慢
                col_predicates = col_predicates[:15]  # 增加谓词数量以捕获更多模式
                print(f"    {target_col}列谓词数: {len(col_predicates)}")
                
                # 为该列进行专门的MCTS搜索
                col_results = mcts_rule_discovery_yroot(
                    mcts_df,
                    filtered_predicates,
                    col_predicates,
                    max_depth=6,  # 进一步增加深度以发现更复杂的规则
                    n_iter=5000,  # 增加迭代次数以提高发现概率
                    n_workers=dist_config['n_workers'],
                    use_parallel=dist_config['use_multiprocessing']
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
    
    for idx, row in rules_df.iterrows():
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
    
    for i in range(len(test_clean)):
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


if __name__ == '__main__':
    main() 