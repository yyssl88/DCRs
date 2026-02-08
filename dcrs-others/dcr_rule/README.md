# Multimodal DCRLearner Pipeline

## 功能简介
本项目实现了一个多模态数据规则发现与查错的自动化pipeline，支持文本、图片等多模态特征的embedding提取、规则自动生成、基于MCTS的规则发现、查错与评估。

## 主要特性
- 自动识别表格字段类型（文本、枚举、数值、图片路径）
- 文本特征用BERT/BGE-M3等模型提取embedding
- 图片特征用Qwen等多模态模型提取embedding和类别
- 支持embedding聚类、特征降维
- 自动生成谓词（predicate）并构造规则
- 基于MCTS（蒙特卡罗树搜索）的分布式的高效规则发现
- 支持规则查错、与真实数据对比评估F1/Recall/Precision

## 效果
#### amazon数据集

precision: 1 recall:0.7 **F1:0.82**

#### goodreads数据集

precision:1 recall:0.82  **F1:0.90** 

#### fakeddit数据集

 precision:1 recall:0.89 **F1:0.94**

## 数据准备
- 数据目录结构示例：
  ```
  data_dir/
    train_clean.csv
    test_clean.csv
    test_dirty.csv
    imgs/  # 图片文件夹
  ```
- train_clean.csv/test_clean.csv/test_dirty.csv：表格字段需包含文本、枚举、数值、图片路径等。
- imgs/：图片文件夹，img_path列需能拼接出图片完整路径。

## 主要流程
1. **数据读取与类型识别**
   - 自动识别每列类型（文本、枚举、数值、img_path）。
2. **特征提取**
   - 文本列用BERT/BGE-M3提取embedding。
   - 图片列用Qwen等模型提取embedding和类别。
3. **特征聚类与降维**
   - 对embedding做PCA降维、层次聚类，生成聚类标签。
4. **谓词生成**
   - 自动为枚举/数值型字段生成谓词（=, !=, >, <, ...）。
5. **规则发现（MCTS）**
   - 支持度筛选谓词，减少搜索空间。
   - 以每个枚举型谓词为Y predicate，MCTS搜索最佳前提组合。
   - 输出规则表dcr_mcts_rule.csv（y_pred, best_rule, support, confidence）。
6. **规则查错**
   - 用规则在test_dirty.csv查找异常单元格（error_cell: 行号, 列名）。
   - 输出查错结果dcr_rule_error_detect.csv。
7. **查错评估**
   - 与test_clean.csv对比，自动计算F1、Recall、Precision。
   - 输出评估指标dcr_rule_error_metrics.txt。

## 输出说明
- `train_with_embeddings.pkl`：带embedding的训练数据。
- `train_extend.csv`：扩展特征表（去除高维embedding）。
- `predicates.txt`：自动生成的谓词列表。
- `dcr_mcts_rule.csv`：MCTS发现的高置信规则表。
- `dcr_rule_error_detect.csv`：规则查错结果（异常单元格）。
- `dcr_rule_error_metrics.txt`：查错评估指标。

 