#!/usr/bin/env python3
"""
多数据集精确噪音控制脚本
支持 Amazon、FakeDDIT、Goodreads 三个数据集
包含图片对应检查和验证功能
"""

import os
import random
import pandas as pd
import numpy as np
import shutil
import json
from PIL import Image

# ================== 配置区 ==================
SEED = 42
NOISE_RATIO = 0.1  # 10%噪音比例（按行数比例）
NOISE_CELLS = 100  # 精确控制噪音单元格数量（优先级高于NOISE_RATIO）
USE_EXACT_CELL_COUNT = True  # 是否使用精确单元格数量控制

# 多数据集配置
DATASET_CONFIG = {
    'amazon': {
        'data_dir': '/data_nas/DCR/data_her/data/amazon',
        'relation_file': 'amazon_com_best_sellers_2025_01_27.csv',
        'img_dir': 'imgs',
        'sep': ',',
        'output_dir': '/data_nas/DCR/split_addnoise/amazon_test',
        'best_img_dict': 'best_img_dict.json',
        'her_map': 'amazon_her_map.json',
    },
    'fakeddit': {
        'data_dir': '/data_nas/DCR/data_her/data/fakeddit',
        'relation_file': 'all_train.tsv',
        'img_dir': 'imgs',
        'sep': '\t',
        'output_dir': '/data_nas/DCR/split_addnoise/fakeddit_test',
        'best_img_dict': 'best_img_dict.json',
        'her_map': 'fakeddit_her_map.json',
    },
    'goodreads': {
        'data_dir': '/data_nas/DCR/data_her/data/goodreads',
        'relation_file': 'GoodReads_100k_books.csv',
        'img_dir': 'imgs',
        'sep': ',',
        'output_dir': '/data_nas/DCR/split_addnoise/goodreads_test',
        'best_img_dict': 'best_img_dict.json',
        'her_map': 'goodreads_her_map.json',
    },
    'ml25m': {
        'data_dir': '/data_nas/DCR/data_her/data/ml25m',
        'relation_file': 'movie_wide_table.csv',
        'img_dir': 'covers/covers/',
        'sep': ',',
        'output_dir': '/data_nas/DCR/split_addnoise/ml25m_test',
        'best_img_dict': 'best_img_dict.json',
        'her_map': 'ml25m_her_map.json',
    },
}

# ================ 辅助函数 =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def split_df(df, ratios=(0.8, 0.1, 0.1), seed=42):
    """拆分数据集为训练、验证、测试集"""
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    n = len(df)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]
    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]

def add_precise_noise(df, noise_ratio=0.1, noise_cells=None, use_exact_cell_count=False, seed=42):
    """
    精确控制噪音添加：支持按行数比例或精确单元格数量添加噪音
    排除img_path列和文本描述列
    
    Args:
        df: 输入数据框
        noise_ratio: 按行数比例添加噪音（当use_exact_cell_count=False时使用）
        noise_cells: 精确的噪音单元格数量（当use_exact_cell_count=True时使用）
        use_exact_cell_count: 是否使用精确单元格数量控制
        seed: 随机种子
    """
    noisy_df = df.copy()
    n = len(df)
    
    print(f"📊 数据集信息:")
    print(f"  - 总行数: {n}")
    print(f"  - 总列数: {len(df.columns)}")
    
    if use_exact_cell_count and noise_cells is not None:
        print(f"  - 噪音控制模式: 精确单元格数量")
        print(f"  - 目标噪音单元格数: {noise_cells}")
        print(f"  - 噪音类型: 仅枚举型列")
    else:
        n_noise_rows = max(1, int(n * noise_ratio))  # 至少1行
        print(f"  - 噪音控制模式: 按行数比例")
        print(f"  - 噪音行数: {n_noise_rows} ({noise_ratio*100:.1f}%)")
        print(f"  - 噪音类型: 仅枚举型列")
    
    # 定义需要排除的列
    excluded_cols = {
        'img_path',  # 图片路径列
        'description', 'descriptionRaw', 'name', 'title',  # 文本描述列
        'features', 'breadcrumbs', 'additionalProperties',  # 详细描述列
        'url', 'imageUrls', 'variants',  # URL和图片相关列
        'scrapedDate', 'new_path', 'nodeName',  # 元数据列
        'gtin', 'mpn', 'sku', 'style'  # 产品标识列
    }
    
    # 识别数值型和枚举类型列（排除指定列）
    numeric_cols = []
    enum_cols = []
    
    for col in noisy_df.columns:
        # 跳过排除的列
        if col in excluded_cols:
            continue
            
        if pd.api.types.is_numeric_dtype(noisy_df[col]):
            numeric_cols.append(col)
        elif noisy_df[col].dtype == 'object':
            # 检查是否为枚举类型（唯一值数量相对较少）
            unique_ratio = noisy_df[col].nunique() / len(noisy_df[col].dropna())
            if unique_ratio < 0.5 and noisy_df[col].nunique() > 1:  # 唯一值比例小于50%且多于1个
                enum_cols.append(col)
    
    print(f"🔧 列类型识别:")
    print(f"  - 排除的列: {sorted(excluded_cols.intersection(set(df.columns)))}")
    print(f"  - 数值型列 ({len(numeric_cols)}): {numeric_cols} (不添加噪音)")
    print(f"  - 枚举型列 ({len(enum_cols)}): {enum_cols} (添加噪音)")
    print(f"  - 其他列 ({len(df.columns) - len(numeric_cols) - len(enum_cols) - len(excluded_cols.intersection(set(df.columns)))}): {[col for col in df.columns if col not in numeric_cols and col not in enum_cols and col not in excluded_cols]}")
    
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 记录修改统计
    modification_stats = {
        'selected_rows': set(),
        'modified_rows': set(),
        'total_changes': 0,
        'changes_by_column': {}
    }
    
    if use_exact_cell_count and noise_cells is not None:
        # 精确单元格数量模式
        print(f"🔧 精确单元格数量模式: 目标 {noise_cells} 个单元格")
        
        # 创建所有可修改的单元格列表（只包含枚举型列）
        modifiable_cells = []
        for row_idx in noisy_df.index:
            for col in enum_cols:  # 只对枚举型列添加噪音
                if pd.notna(noisy_df.at[row_idx, col]):
                    modifiable_cells.append((row_idx, col))
        
        print(f"📊 可修改的单元格总数: {len(modifiable_cells)}")
        
        if len(modifiable_cells) < noise_cells:
            print(f"⚠️  可修改单元格数量 ({len(modifiable_cells)}) 少于目标数量 ({noise_cells})")
            noise_cells = len(modifiable_cells)
            print(f"📊 调整为实际可修改数量: {noise_cells}")
        
        # 随机选择要修改的单元格
        selected_cells = random.sample(modifiable_cells, noise_cells)
        print(f"📊 选中的单元格数量: {len(selected_cells)}")
        
        # 对选中的单元格添加噪音（只处理枚举型列）
        for i, (row_idx, col) in enumerate(selected_cells):
            if i % 10 == 0:  # 每10个单元格打印一次进度
                print(f"🔧 处理进度: {i+1}/{len(selected_cells)}")
            
            original_val = noisy_df.at[row_idx, col]
            
            # 只处理枚举型列
            if col in enum_cols:
                # 枚举型列处理
                # 获取该列的所有唯一值
                unique_values = noisy_df[col].dropna().unique().tolist()
                
                if len(unique_values) > 1:
                    # 随机选择其他值
                    other_values = [v for v in unique_values if v != original_val]
                    if other_values:
                        new_val = random.choice(other_values)
                        noisy_df.at[row_idx, col] = new_val
                        print(f"  📊 枚举列 {col}: {original_val} -> {new_val}")
                    else:
                        continue  # 如果没有其他值可选，跳过
                else:
                    continue  # 如果只有一个唯一值，跳过
            else:
                # 如果不是枚举型列，跳过
                continue
            
            # 记录统计信息
            modification_stats['selected_rows'].add(row_idx)
            modification_stats['modified_rows'].add(row_idx)
            modification_stats['total_changes'] += 1
            
            if col not in modification_stats['changes_by_column']:
                modification_stats['changes_by_column'][col] = 0
            modification_stats['changes_by_column'][col] += 1
        
        print(f"🔧 处理完成: {len(selected_cells)} 个单元格")
        
    else:
        # 按行数比例模式（只对枚举型列添加噪音）
        n_noise_rows = max(1, int(n * noise_ratio))  # 至少1行
        noise_row_indices = np.random.choice(noisy_df.index, n_noise_rows, replace=False)
        print(f"📊 选中的行索引: {sorted(noise_row_indices.tolist())}")
        
        modification_stats['selected_rows'] = set(noise_row_indices)
        
        # 对选中的行添加噪音（只对枚举型列）
        for row_idx in noise_row_indices:
            print(f"\n🔧 处理行 {row_idx}:")
            row_changes = 0
            
            # 只对枚举型列添加噪音
            for col in enum_cols:
                if pd.notna(noisy_df.at[row_idx, col]):
                    # 获取该列的所有唯一值
                    unique_values = noisy_df[col].dropna().unique().tolist()
                    
                    if len(unique_values) > 1:
                        original_val = noisy_df.at[row_idx, col]
                        
                        # 随机选择其他值
                        other_values = [v for v in unique_values if v != original_val]
                        if other_values:
                            new_val = random.choice(other_values)
                            noisy_df.at[row_idx, col] = new_val
                            print(f"  📊 枚举列 {col}: {original_val} -> {new_val}")
                            row_changes += 1
                            
                            # 记录列变化统计
                            if col not in modification_stats['changes_by_column']:
                                modification_stats['changes_by_column'][col] = 0
                            modification_stats['changes_by_column'][col] += 1
            
            if row_changes > 0:
                modification_stats['modified_rows'].add(row_idx)
                modification_stats['total_changes'] += row_changes
                print(f"  📈 行 {row_idx} 总计修改: {row_changes} 个单元格")
    
    # 详细统计报告
    print(f"\n📊 详细统计报告:")
    print(f"  - 选中的行数: {len(modification_stats['selected_rows'])}")
    print(f"  - 实际修改的行数: {len(modification_stats['modified_rows'])}")
    print(f"  - 总修改单元格数: {modification_stats['total_changes']}")
    
    if len(modification_stats['selected_rows']) > 0:
        print(f"  - 平均每行修改: {modification_stats['total_changes']/len(modification_stats['selected_rows']):.1f} 个单元格")
    
    print(f"\n📊 各列修改统计:")
    for col, changes in modification_stats['changes_by_column'].items():
        print(f"  - {col}: {changes} 个变化")
    
    # 验证修改比例
    total_cells = len(df) * len(df.columns)
    change_ratio = modification_stats['total_changes'] / total_cells * 100
    print(f"\n📊 总体修改比例:")
    print(f"  - 修改单元格比例: {change_ratio:.2f}%")
    print(f"  - 修改行比例: {len(modification_stats['modified_rows'])/n*100:.1f}%")
    
    if use_exact_cell_count and noise_cells is not None:
        print(f"  - 目标单元格数: {noise_cells}")
        print(f"  - 实际修改单元格数: {modification_stats['total_changes']}")
        if modification_stats['total_changes'] == noise_cells:
            print(f"  ✅ 精确达到目标单元格数量")
        else:
            print(f"  ⚠️  实际修改数量与目标数量不符")
    
    return noisy_df

def load_best_img_dict(cfg):
    """加载best_img_dict.json文件"""
    best_img_path = os.path.join(cfg['data_dir'], cfg['best_img_dict'])
    if not os.path.exists(best_img_path):
        print(f"❌ best_img_dict文件不存在: {best_img_path}")
        return None
    
    try:
        with open(best_img_path, 'r', encoding='utf-8') as f:
            best_img_dict = json.load(f)
        print(f"✅ 成功加载best_img_dict: {len(best_img_dict)} 个有效样本")
        return best_img_dict
    except Exception as e:
        print(f"❌ 加载best_img_dict失败: {e}")
        return None

def filter_data_by_best_img(df, best_img_dict):
    """根据best_img_dict过滤数据并添加图片信息"""
    if best_img_dict is None:
        return df
    
    # 获取有效的行号（best_img_dict的key）
    valid_indices = list(best_img_dict.keys())
    # 转换为整数索引
    valid_indices = [int(idx) for idx in valid_indices]
    
    # 过滤掉超出数据范围的索引
    max_valid_index = len(df) - 1
    valid_indices = [idx for idx in valid_indices if 0 <= idx <= max_valid_index]
    
    if len(valid_indices) == 0:
        print(f"⚠️  所有索引都超出数据范围，使用原始数据")
        return df
    
    print(f"📊 索引过滤: 原始索引数 {len(list(best_img_dict.keys()))}, 有效索引数 {len(valid_indices)}")
    
    # 过滤数据
    filtered_df = df.iloc[valid_indices].copy()
    
    # 添加图片路径信息（只对应有效的索引）
    img_paths = [best_img_dict[str(idx)] for idx in valid_indices]
    filtered_df['img_path'] = img_paths
    
    print(f"📊 过滤后样本数: {len(filtered_df)} (原始: {len(df)})")
    return filtered_df

def check_image_correspondence(df, cfg):
    """检查图片对应是否正确"""
    print("🔍 检查图片对应关系...")
    
    if 'img_path' not in df.columns:
        print("⚠️  数据中没有img_path列")
        return False
    
    img_dir = os.path.join(cfg['data_dir'], cfg['img_dir'])
    if not os.path.exists(img_dir):
        print(f"❌ 图片目录不存在: {img_dir}")
        return False
    
    # 统计信息
    total_images = len(df)
    existing_images = 0
    missing_images = 0
    corrupted_images = 0
    valid_images = 0
    
    print(f"📁 图片目录: {img_dir}")
    print(f"📊 检查 {total_images} 个图片路径...")
    
    for idx, row in df.iterrows():
        img_path = row['img_path']
        if pd.isna(img_path):
            missing_images += 1
            continue
        
        # 构建完整路径
        full_img_path = os.path.join(img_dir, img_path)
        
        if not os.path.exists(full_img_path):
            missing_images += 1
            print(f"  ❌ 图片不存在: {img_path}")
            continue
        
        existing_images += 1
        
        # 尝试打开图片验证完整性
        try:
            with Image.open(full_img_path) as img:
                img.verify()  # 验证图片完整性
            valid_images += 1
        except Exception as e:
            corrupted_images += 1
            print(f"  ⚠️  图片损坏: {img_path} - {e}")
    
    # 输出统计结果
    print(f"\n📊 图片对应检查结果:")
    print(f"  - 总图片数: {total_images}")
    print(f"  - 存在图片: {existing_images} ({existing_images/total_images*100:.1f}%)")
    print(f"  - 缺失图片: {missing_images} ({missing_images/total_images*100:.1f}%)")
    print(f"  - 损坏图片: {corrupted_images} ({corrupted_images/total_images*100:.1f}%)")
    print(f"  - 有效图片: {valid_images} ({valid_images/total_images*100:.1f}%)")
    
    # 判断是否通过检查
    success_rate = valid_images / total_images
    if success_rate >= 0.9:  # 90%以上图片有效
        print(f"✅ 图片对应检查通过 (成功率: {success_rate*100:.1f}%)")
        return True
    elif success_rate >= 0.7:  # 70%以上图片有效
        print(f"⚠️  图片对应检查警告 (成功率: {success_rate*100:.1f}%)")
        return True
    else:
        print(f"❌ 图片对应检查失败 (成功率: {success_rate*100:.1f}%)")
        return False

def process_dataset(dataset_name, cfg):
    """处理单个数据集"""
    print(f"\n{'='*50}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*50}")
    
    data_path = os.path.join(cfg['data_dir'], cfg['relation_file'])
    sep = cfg['sep']
    out_dir = cfg['output_dir']
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")
    
    # 加载best_img_dict
    best_img_dict = load_best_img_dict(cfg)
    
    print(f"📁 读取数据: {data_path}")
    try:
        # 尝试不同的编码格式
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(data_path, sep=sep, encoding=encoding)
                print(f"✅ 成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"❌ 所有编码格式都无法读取文件")
            return False
            
        print(f"📊 原始样本数: {len(df)}")
        print(f"📊 原始列数: {len(df.columns)}")
        print(f"📊 原始列名: {list(df.columns)}")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return False
    
    # 根据best_img_dict过滤数据
    df = filter_data_by_best_img(df, best_img_dict)
    
    # 检查图片对应关系
    if not check_image_correspondence(df, cfg):
        print("⚠️  图片对应检查失败，但继续处理...")

    # 拆分
    train_df, valid_df, test_df = split_df(df, seed=SEED)
    print(f"📈 拆分结果 - train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")
    
    # 复制HER匹配结果文件（如果存在）
    her_map_path = os.path.join(cfg['data_dir'], cfg['her_map'])
    if os.path.exists(her_map_path):
        her_map_dst = os.path.join(out_dir, cfg['her_map'])
        try:
            # 直接读取内容并写入，避免权限问题
            with open(her_map_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(her_map_dst, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已复制HER匹配结果: {cfg['her_map']}")
        except Exception as e:
            print(f"❌ 复制HER匹配结果失败: {e}")
    else:
        print(f"⚠️  HER匹配结果文件不存在: {her_map_path}")
    
    # 保存clean数据集
    def save_csv_with_retry(df, filepath, max_retries=3):
        """带重试机制的CSV保存"""
        for attempt in range(max_retries):
            try:
                df.to_csv(filepath, index=False)
                return True
            except OSError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  保存失败，重试 {attempt + 1}/{max_retries}: {filepath}")
                    import time
                    time.sleep(1)  # 等待1秒后重试
                else:
                    print(f"❌ 保存失败: {filepath}, 错误: {e}")
                    return False
        return False
    
    # 保存clean数据集
    save_success = True
    save_success &= save_csv_with_retry(train_df, os.path.join(out_dir, 'train_clean.csv'))
    save_success &= save_csv_with_retry(valid_df, os.path.join(out_dir, 'valid_clean.csv'))
    save_success &= save_csv_with_retry(test_df, os.path.join(out_dir, 'test_clean.csv'))
    
    if save_success:
        print("✅ 已保存clean数据集")
    else:
        print("❌ 保存clean数据集失败")
        return False

    # valid/test加噪音
    print("\n🔧 为valid数据集添加噪音...")
    if USE_EXACT_CELL_COUNT:
        valid_dirty = add_precise_noise(valid_df, noise_cells=NOISE_CELLS, use_exact_cell_count=True, seed=SEED)
    else:
        valid_dirty = add_precise_noise(valid_df, noise_ratio=NOISE_RATIO, seed=SEED)
    
    print("\n🔧 为test数据集添加噪音...")
    if USE_EXACT_CELL_COUNT:
        test_dirty = add_precise_noise(test_df, noise_cells=NOISE_CELLS, use_exact_cell_count=True, seed=SEED)
    else:
        test_dirty = add_precise_noise(test_df, noise_ratio=NOISE_RATIO, seed=SEED)
    
    # 保存dirty数据集
    save_success = True
    save_success &= save_csv_with_retry(valid_dirty, os.path.join(out_dir, 'valid_dirty.csv'))
    save_success &= save_csv_with_retry(test_dirty, os.path.join(out_dir, 'test_dirty.csv'))
    
    if save_success:
        print("✅ 已保存dirty数据集")
    else:
        print("❌ 保存dirty数据集失败")
        return False
    
    # 复制图片文件夹
    img_src_dir = os.path.join(cfg['data_dir'], cfg['img_dir'])
    img_dst_dir = os.path.join(out_dir, cfg['img_dir'])
    if os.path.exists(img_src_dir):
        if os.path.exists(img_dst_dir):
            shutil.rmtree(img_dst_dir)  # 如果目标目录存在，先删除
        try:
            # 使用自定义复制函数，避免权限问题
            def copy_tree_without_metadata(src, dst):
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for item in os.listdir(src):
                    s = os.path.join(src, item)
                    d = os.path.join(dst, item)
                    if os.path.isdir(s):
                        copy_tree_without_metadata(s, d)
                    else:
                        # 直接读取内容并写入，避免权限问题
                        try:
                            with open(s, 'rb') as f_src:
                                content = f_src.read()
                            with open(d, 'wb') as f_dst:
                                f_dst.write(content)
                        except Exception as e:
                            print(f"⚠️  复制文件失败 {s}: {e}")
            
            copy_tree_without_metadata(img_src_dir, img_dst_dir)
            print(f"✅ 已复制图片文件夹: {img_src_dir} -> {img_dst_dir}")
        except Exception as e:
            print(f"❌ 复制图片文件夹失败: {e}")
    else:
        print(f"⚠️  图片文件夹不存在: {img_src_dir}")
    
    # 打印clean和dirty数据的前5行
    print("\n📊 Clean数据前5行:")
    print("=" * 50)
    print("Train Clean:")
    print(train_df.head())
    print("\nValid Clean:")
    print(valid_df.head())
    print("\nTest Clean:")
    print(test_df.head())
    
    print("\n📊 Dirty数据前5行:")
    print("=" * 50)
    print("Valid Dirty:")
    print(valid_dirty.head())
    print("\nTest Dirty:")
    print(test_dirty.head())
    
    return True

def main():
    """主函数 - 处理所有数据集"""
    set_seed(SEED)
    print("🚀 开始处理所有数据集")
    if USE_EXACT_CELL_COUNT:
        print(f"🔧 配置 - SEED: {SEED}, NOISE_CELLS: {NOISE_CELLS} (精确单元格数量模式)")
    else:
        print(f"🔧 配置 - SEED: {SEED}, NOISE_RATIO: {NOISE_RATIO} (按行数比例模式)")
    
    success_count = 0
    total_count = len(DATASET_CONFIG)
    
    for dataset_name, cfg in DATASET_CONFIG.items():
        if process_dataset(dataset_name, cfg):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"🎉 处理完成! 成功: {success_count}/{total_count}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main() 