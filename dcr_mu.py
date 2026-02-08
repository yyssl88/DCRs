import os
import json
import re
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import hashlib
from collections import deque, defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# =========================================================
# 1. Configuration (Paper Section 5)
# =========================================================
class DCRConfig:
    # Paths
    BASE_DIR = "/data_nas/DCR/split_addnoise/pad_mu"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MU_NOISE_DIR = os.path.join(DATA_DIR, "outputs", "mu_noise")
    FIXED_CLEAN_MR_PATH = "/data_nas/DCR/split_addnoise/pad_mu/data/mr_outputs_cluster3/noise_00/mr_metadata_train_augmented.csv"
    TEST_MR_DIR = "/data_nas/DCR/split_addnoise/pad_mu/data/mr_outputs_cluster3/noise_00"
    OUT_DIR = os.path.join(DATA_DIR, "dcr_final_results222")
    
    # Discovery Parameters
    # Sigma: Paper uses 1e-6 * |D|^2 for count, or just probability.
    # Since we strictly use [0,1] probability for support now:
    # If |D|=1000, 1 pair is 1/10^6. So threshold should be around 1e-6 to 1e-4.
    SUPPORT_THRESHOLD = 0.001   # Sigma (Probability, strictly [0,1])
    CONF_THRESHOLD = 0.8        # Delta (Probability, strictly [0,1])
    MAX_DEPTH = 4               # Eta
    
    # MCTS Parameters
    OUTER_ITERATIONS = 7
    MCTS_ITERATIONS = 1000
    C_PARAM = 1.5              # c (Exploration weight)
    ALPHA = 0.8                # Balancing parameter
    
    # Training
    HIDDEN_DIM = 512
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 256
    MEMORY_SIZE = 10000
    SAMPLE_BATCH_SIZE = 2000
    
    TARGET_COLS = ["diagnostic"]

# =========================================================
# 2. Predicate Logic (Section 2.1 & Pruning)
# =========================================================
class Predicate:
    def __init__(self, type_, col, op, val=None, other_col=None, alias=None):
        self.type = type_   # 'unary' or 'binary'
        self.col = col
        self.op = op        # '=', '!='
        self.val = val      # Constant c (for unary)
        self.other_col = other_col # Attribute B (for binary)
        self.alias = alias  # 't1' or 't2' (Only for unary in binary rules)

    def __str__(self):
        if self.type == 'unary':
            # Strict Tuple Alias: t1.A = c vs t2.A = c
            prefix = self.alias if self.alias else "t"
            return f"{prefix}.{self.col} {self.op} '{self.val}'"
        else:
            # Binary is always t1 vs t2 in DCR context
            return f"t1.{self.col} {self.op} t2.{self.other_col}"

    def __repr__(self): return self.__str__()
    def __eq__(self, other): return str(self) == str(other)
    def __hash__(self): return hash(str(self))
    
    def to_json(self):
        return {"type": self.type, "col": self.col, "op": self.op, 
                "val": str(self.val), "other_col": self.other_col, "alias": self.alias}
    
def is_valid_expansion(current_preds, new_pred, consequence=None):
    if new_pred in current_preds:
        return False
    
    knowledge_eq = {'t1': {}, 't2': {}}
    knowledge_range = {'t1': {}, 't2': {}}
    binary_relations = {} # (col1, col2) -> op
    
    all_preds = current_preds + [new_pred]
    
    # 1. Build Knowledge Base
    for p in all_preds:
        if p.type == 'unary':
            a = p.alias
            col = p.col
            op = p.op
            val = p.val
            if op == '==':
                if col in knowledge_eq[a]:
                    if knowledge_eq[a][col] != val: return False # Unary Contradiction
                knowledge_eq[a][col] = val
            
            # Range check (simplified)
            if op in ['>', '<='] and isinstance(val, (int, float)):
                if col not in knowledge_range[a]: knowledge_range[a][col] = []
                for exist_op, exist_val in knowledge_range[a][col]:
                    if op == '>' and exist_op == '<=' and val >= exist_val - 1e-6: return False
                    if op == '<=' and exist_op == '>' and val <= exist_val + 1e-6: return False
                knowledge_range[a][col].append((op, val))
        
        elif p.type == 'binary':
            # Store binary relations to check for direct contradictions
            key = tuple(sorted((p.col, p.other_col)))
            if key not in binary_relations: binary_relations[key] = set()
            
            # Check Direct Binary Contradictions: A==B vs A!=B
            if p.op == '==' and '!=' in binary_relations[key]: return False
            if p.op == '!=' and '==' in binary_relations[key]: return False
            binary_relations[key].add(p.op)

    # 2. Advanced Check: Binary vs Unary Implications
    for p in all_preds:
        if p.type == 'binary':
            c1, c2 = p.col, p.other_col
            op = p.op
            
            v1 = knowledge_eq['t1'].get(c1)
            v2 = knowledge_eq['t2'].get(c2)
            
            # Case A: We know both values (v1, v2)
            if v1 is not None and v2 is not None:
                # Contradiction Check
                if op == '==' and v1 != v2: return False
                if op == '!=' and v1 == v2: return False
                if op == '>' and not (v1 > v2): return False # Assuming comparable
                if op == '<=' and not (v1 <= v2): return False
                
                # Redundancy Check (Pruning):
                # If we know t1.A=x and t2.B=y, and x=y, then adding t1.A == t2.B is redundant.
                # The rule is "implied" by Unary parts.
                # However, strictly speaking, redundancy doesn't make it invalid, just useless.
                # But to save search depth, we should prune it.
                if op == '==' and v1 == v2: return False # Implied
                if op == '!=' and v1 != v2: return False # Implied
            
            # Case B: We know A == B (from Binary), check Unary Consistency
            if op == '==':
                # If t1.A == t2.B, and we know t1.A=x, then implied t2.B=x
                if v1 is not None and v2 is None:
                    # Potential future check: if adding t2.B=v1 causes Unary conflict?
                    pass 
    
    # 3. Binary Predicate Requirement ---
    # Logic: If the rule contains both t1 and t2 unary predicates, it must have at least one binary predicate.
    # This ensures that t1 and t2 are connected by a relationship.
    has_t1_unary = any(p.type == 'unary' and p.alias == 't1' for p in all_preds)
    has_t2_unary = any(p.type == 'unary' and p.alias == 't2' for p in all_preds)
    has_binary = any(p.type == 'binary' for p in all_preds)
    
    if has_t1_unary and has_t2_unary and not has_binary:
        return False
    
    # 4. Consequence-based Unary Predicate Restriction ---
    # Logic: If consequence is a t1 unary predicate and all premise predicates are unary,
    # then all premise unary predicates must also be t1.
    if consequence and consequence.type == 'unary' and consequence.alias == 't1':
        all_premise_unary = all(p.type == 'unary' for p in all_preds)
        if all_premise_unary:
            has_t2_in_premise = any(p.type == 'unary' and p.alias == 't2' for p in all_preds)
            if has_t2_in_premise:
                return False
                
    return True

# =========================================================
# 3. Neural Network (Section 5.3)
# =========================================================
class DCRNeuralAgent(nn.Module):
    def __init__(self, action_space_size, hidden_dim=128):
        super(DCRNeuralAgent, self).__init__()
        self.action_space_size = action_space_size
        
        # State Encoder: Bag-of-Predicates -> Dense Vector
        self.encoder = nn.Sequential(
            nn.Linear(action_space_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy Head (Mp): Logits for next action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )
        
        # Value Head (Mv): Expected Reward [0, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        policy_logits = self.policy_head(h)
        value_est = self.value_head(h)
        return policy_logits, value_est

# =========================================================
# 4. Metrics Calculation (Section 5.1)
# =========================================================
def calculate_metrics(df, premise_preds, consequence_pred):
    """
    Exact O(N^2) Enumeration using Matrix Broadcasting.
    Implementation of Definition 5 (C1) and Definition 6 (C2).
    Returns Probabilities in [0, 1].
    """
    N = len(df)
    if N < 2: return 0.0, 0.0 # Cannot compute pairs with < 2 rows
    
    # Determine Rule Type based on predicates
    is_binary = any(p.type == 'binary' for p in premise_preds) or (consequence_pred.type == 'binary')
    
    # Helper to get mask for a single predicate
    def get_unary_mask(col_series, op, val):
        if op == '==': return (col_series.astype(str) == str(val)).values
        elif op == '!=': return (col_series.astype(str) != str(val)).values
        # Numerical
        elif op == '>': return (col_series > val).values
        elif op == '<=': return (col_series <= val).values
        return np.ones(len(col_series), dtype=bool)

    # Helper for binary
    def get_binary_matrix(v1_vec, v2_vec, op):
        if op == '==': return (v1_vec == v2_vec)
        elif op == '!=': return (v1_vec != v2_vec)
        elif op == '>': return (v1_vec > v2_vec)
        elif op == '<=': return (v1_vec <= v2_vec)
        return np.ones((len(v1_vec), len(v2_vec)), dtype=bool)    
    
    if not is_binary:
        # --- Unary [C2]: Single-Tuple Constraint ---
        # Semantics: Ignore 't1'/'t2' aliases. Treat everything as generic 't'.
        # Formula: |{t | t |= X ^ p0}| / |D|
        
        mask = np.ones(N, dtype=bool)
        for p in premise_preds:
            mask &= get_unary_mask(df[p.col], p.op, p.val)
        
        supp_X_count = mask.sum()
        if supp_X_count == 0: return 0.0, 0.0
        
        p0 = consequence_pred
        cons_mask = get_unary_mask(df[p0.col], p0.op, p0.val)
        
        supp_rule_count = (mask & cons_mask).sum()
        # Metric: Probability in D
        support = supp_rule_count / N 
        confidence = supp_rule_count / supp_X_count
        return float(support), float(confidence)

    else:
        # --- Binary [C1]: Pairwise Constraint ---
        # Semantics: Strictly respect 't1' and 't2' aliases.
        # Formula: |{(t1,t2) | t1!=t2 ^ ...}| / (|D|*(|D|-1))
        
        mask_t1 = np.ones(N, dtype=bool)
        mask_t2 = np.ones(N, dtype=bool)
        binary_matrices = [] 
        
        for p in premise_preds:
            if p.type == 'unary':
                p_mask = get_unary_mask(df[p.col], p.op, p.val)
                # ALIAS ENFORCEMENT
                if p.alias == 't1':
                    mask_t1 &= p_mask
                elif p.alias == 't2':
                    mask_t2 &= p_mask
                # Note: If alias is missing in binary mode, it's ambiguous. 
                # Our generator ensures alias is set.
                    
            elif p.type == 'binary':
                # Numerical sensitive extraction
                # If op is numeric, keep values; else cast string
                if p.op in ['>', '<=']:
                    v1 = df[p.col].values[:, None]
                    v2 = df[p.other_col].values[None, :]
                else:
                    v1 = df[p.col].astype(str).values[:, None]
                    v2 = df[p.other_col].astype(str).values[None, :]
                
                binary_matrices.append(get_binary_matrix(v1, v2, p.op))
        
        # Combine Unary Masks -> (N, N) Matrix
        # P[i,j] is True if t_i satisfies t1-preds AND t_j satisfies t2-preds
        premise_matrix = mask_t1[:, None] & mask_t2[None, :]
        
        for b_mat in binary_matrices:
            premise_matrix &= b_mat
            
        # --- CRITICAL FIX: Exclude Self-Pairs (Diagonal) ---
        # We strictly require t1 != t2 for DC/DCR discovery
        np.fill_diagonal(premise_matrix, False)
        supp_X_count = premise_matrix.sum()
        if supp_X_count == 0: return 0.0, 0.0
        
        # Consequence Matrix
        p0 = consequence_pred
        if p0.type == 'unary':
            p0_mask = get_unary_mask(df[p0.col], p0.op, p0.val)
            if p0.alias == 't1':
                consequence_matrix = p0_mask[:, None] & np.ones((1, N), dtype=bool)
            else: # t2
                consequence_matrix = np.ones((N, 1), dtype=bool) & p0_mask[None, :]
        else:
            if p0.op in ['>', '<=']:
                v1 = df[p0.col].values[:, None]
                v2 = df[p0.other_col].values[None, :]
            else:
                v1 = df[p0.col].astype(str).values[:, None]
                v2 = df[p0.other_col].astype(str).values[None, :]
            consequence_matrix = get_binary_matrix(v1, v2, p0.op)
        
        # Apply Diagonal Mask to Consequence as well (consistency)
        np.fill_diagonal(consequence_matrix, False)
        rule_matrix = premise_matrix & consequence_matrix
        supp_rule_count = rule_matrix.sum()

        # Total valid pairs = N * (N - 1)
        total_pairs = N * (N - 1)
        
        support = supp_rule_count / total_pairs
        confidence = supp_rule_count / supp_X_count
        
        return float(support), float(confidence)

# =========================================================
# 5. MCTS Logic with Trajectory Collection
# =========================================================
class MCTSNode:
    def __init__(self, premise, parent=None, action_idx=None):
        self.premise = premise
        self.parent = parent
        self.action_idx = action_idx
        self.children = {}
        
        self.N = 0
        self.Q_emp = 0.0
        self.Q_val = 0.0
        self.P = 0.0

    @property
    def Q(self):
        if self.N == 0: return self.Q_val
        return DCRConfig.ALPHA * self.Q_emp + (1 - DCRConfig.ALPHA) * self.Q_val

def encode_state(premise, action_map, action_size):
    vec = torch.zeros(action_size)
    for p in premise:
        if p in action_map:
            vec[action_map[p]] = 1.0
    return vec

def calculate_reward(support, confidence, premise, consequence):
    """
    Paper Section 5.2.3 (Modified):
    Reward logic:
    1. Hard thresholding for validity.
    2. Differential reward based on predicate type (Constant vs Non-Constant).
    """
    # 1. Basic Validity Check (Hard Threshold)
    if support < DCRConfig.SUPPORT_THRESHOLD or confidence < DCRConfig.CONF_THRESHOLD:
        return 0.0

    # 2. Check for Non-Constant (Binary) Predicates
    # If any predicate in premise or the consequence is 'binary', it's a non-constant rule.
    has_binary = False
    
    if consequence.type == 'binary':
        has_binary = True
    else:
        for p in premise:
            if p.type == 'binary':
                has_binary = True
                break

    # 3. Assign Reward based on type
    if has_binary:
        # for non-constant predicates
        return 0.001 
    else:
        # for constant predicates
        return 1.0

def worker_mcts_search(args):
    """
    Worker function updated with Eq. 2 UCB and Strict Reward.
    """
    task_id, df, p0, all_actions, action_map, agent_state = args
    local_seed = 42 + task_id
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    action_size = len(all_actions)
    
    agent = DCRNeuralAgent(action_size, DCRConfig.HIDDEN_DIM)
    if agent_state: agent.load_state_dict(agent_state)
    agent.eval()
    
    root = MCTSNode(premise=[])
    trajectories = []
    discovered_rules = []
    
    for _ in range(DCRConfig.MCTS_ITERATIONS):
        node = root
        search_path = [node]
        depth = 0
        
        # --- 1. Selection (Corrected UCB Eq. 2) ---
        while True:
            last_idx = node.action_idx if node.action_idx is not None else -1
            potential = [i for i in range(action_size) if i > last_idx]
            FIRST_STEP_COLS = {
                "img_visual_cluster",
                "border_type",
                "surface_type",
                "color_pattern"
            }
            valid_indices = [
                i for i in potential
                if str(all_actions[i]) != str(p0)
                and is_valid_expansion(node.premise, all_actions[i], p0)
                and (
                    node.parent is not None
                    or all_actions[i].col in FIRST_STEP_COLS
                )
            ]

            if not valid_indices or depth >= DCRConfig.MAX_DEPTH:
                break
                
            if len(node.children) < len(valid_indices):
                break
            
            best_child = None
            best_score = -float('inf')
            
            # Sum N(s,a) for all a
            sum_N = sum(child.N for child in node.children.values())
            sqrt_N = math.sqrt(node.N) # Paper says sqrt(N(s))
            
            for idx, child in node.children.items():
                # Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                # Note: node.N is usually sum of visits, which is N(s)
                
                u_term = DCRConfig.C_PARAM * child.P * (sqrt_N / (1 + child.N))
                score = child.Q + u_term
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child:
                node = best_child
                search_path.append(node)
                depth += 1
            else:
                break
        
        # --- 2. Expansion ---
        # (Same as before, logic omitted for brevity, ensure is_valid_expansion is used)
        last_idx = node.action_idx if node.action_idx is not None else -1
        potential = [i for i in range(action_size) if i > last_idx]
        valid_indices = [
            i for i in potential 
            if str(all_actions[i]) != str(p0) and is_valid_expansion(node.premise, all_actions[i], p0)
        ]
        untried = [i for i in valid_indices if i not in node.children]
        
        leaf = node
        if untried:
            state_vec = encode_state(node.premise, action_map, action_size)
            with torch.no_grad():
                logits, val_est = agent(state_vec.unsqueeze(0))
                probs = F.softmax(logits, dim=1)
            
            mask = torch.zeros(action_size)
            mask[untried] = 1.0
            masked_probs = probs[0] * mask
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            else:
                masked_probs[untried] = 1.0 / len(untried)
            
            dist = Categorical(masked_probs)
            action_idx = dist.sample().item()
            action = all_actions[action_idx]
            
            new_child = MCTSNode(premise=node.premise + [action], parent=node, action_idx=action_idx)
            new_child.P = masked_probs[action_idx].item()
            new_child.Q_val = val_est.item()
            
            node.children[action_idx] = new_child
            leaf = new_child
            search_path.append(leaf)

        # --- 3. Simulation & Reward ---
        supp, conf = calculate_metrics(df, leaf.premise, p0)
        # STRICT CHANGE: Hard 0/1 Reward
        reward = calculate_reward(supp, conf, leaf.premise, p0)
        
        if reward > 0.0:
            discovered_rules.append({
                "premise": " ^ ".join([str(p) for p in leaf.premise]),
                "consequence": str(p0),
                "support": supp,
                "confidence": conf
            })
            
        # Store Trajectory
        # STRICT CHANGE: Only store if reward=1 OR weight by reward later?
        # Paper says expectation over M. M usually implies valid rules. 
        # But we also need to train Value network on 0s.
        # We store everything, but Policy Loss will filter.
        if leaf.parent is not None:
            parent_vec = encode_state(leaf.parent.premise, action_map, action_size)
            trajectories.append((parent_vec.cpu().numpy(), leaf.action_idx, reward))

        # --- 4. Back-propagation ---
        for n in reversed(search_path):
            n.N += 1
            n.Q_emp += (reward - n.Q_emp) / n.N
            
    return discovered_rules, trajectories

# =========================================================
# 6. Global Training Loop (Section 5.3)
# =========================================================
def train_agent(agent, memory, optimizer, action_size):
    """
    Strict Policy Gradient (Behavioral Cloning style) without Advantage.
    Strict Value Learning with Binary Targets.
    """
    if len(memory) < DCRConfig.BATCH_SIZE: return 0.0
    
    agent.train()
    total_loss = 0
    
    rng = random.Random(42 + len(memory))
    rng.shuffle(memory)
    batches = [memory[i:i + DCRConfig.BATCH_SIZE] for i in range(0, len(memory), DCRConfig.BATCH_SIZE)]
    
    for batch in batches:
        states, actions, rewards = zip(*batch)
        
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) # 0.0 or 1.0
        
        logits, values = agent(state_tensor)
        
        # 1. Value Loss: MSE against hard 0/1 targets
        # This estimates P(validity | s)
        value_loss = F.mse_loss(values, reward_tensor)
        
        # 2. Policy Loss: Maximize log_prob(a|s) for Valid Trajectories ONLY
        # STRICT CHANGE: No advantage. Filter by reward=1.0
        # Formula: - E_{M} [ log P(a|s) ]
        
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1))
        
        # We effectively mask out samples where reward=0 for the Policy Loss
        # Because "M" in policy formula refers to discovered VALID rules/paths.
        # We use reward as a mask weight.
        policy_loss = -(action_log_probs * reward_tensor).sum() / (reward_tensor.sum() + 1e-6)
        
        # If no valid samples in batch, policy_loss is 0
        
        loss = value_loss + policy_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(batches)

# =========================================================
# 7. Evaluation (Error Detection)
# =========================================================
def is_all_unary_predicate(premise: str) -> bool:
    """判断规则前件是否全部为一元谓词，无二元谓词"""
    binary_pattern = re.compile(r't\d+\.\w+\s*==\s*t\d+\.\w+')
    has_binary = binary_pattern.search(premise) is not None
    return not has_binary

def remove_unary_quote(expr: str, df: pd.DataFrame) -> str:
    """
    智能移除单引号：
    1. 布尔值 ('True'/'False') -> 移除引号
    2. 数值型列 ('50', '3.14') -> 移除引号 (根据 df dtypes 判断)
    3. 字符串/类别列 -> 保留引号
    """
    def replace_quote(match):
        t_attr = match.group(1)  # 例如 t1.age
        op = match.group(2)      # 例如 ==, >, <=
        val_str = match.group(3) # 例如 '50', 'True', 'BCC'
        
        # 获取列名 (移除 t1. / t2. 前缀)
        col_name = t_attr.split('.')[-1]
        
        # 判定是否需要移除引号
        should_remove_quote = False
        
        # 情况A: 布尔值字面量
        if val_str in ('True', 'False'):
            should_remove_quote = True
            
        # 情况B: DataFrame中该列是数值型 (float/int)
        elif df is not None and col_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                # 确保值本身能转成数字，防止意外
                try:
                    float(val_str) 
                    should_remove_quote = True
                except ValueError:
                    pass 
        
        if should_remove_quote:
            return f"{t_attr} {op} {val_str}"
        else:
            return f"{t_attr} {op} '{val_str}'"

    pattern = re.compile(r'(t\d+\.\w+)\s*([!=<>]+)\s*\'([^\']+)\'')
    expr_processed = pattern.sub(replace_quote, expr)
    return expr_processed.replace('^', ' and ')

def extract_consequence_col(consequence: str) -> str:
    """
    从结论谓词中提取列名（适配t1.xxx/t2.xxx/xxx格式）
    """
    col_pattern = re.compile(r'(t\d+\.)?(\w+)\s*==')
    match = col_pattern.search(consequence)
    if match:
        return match.group(2)
    raise ValueError(f"无法从结论谓词中提取列名，结论谓词格式异常：{consequence}")

def find_rule_error_pairs(df_test: pd.DataFrame, rule_dict: dict, df_train: pd.DataFrame = None):
    """
    主方法：自动判断谓词类型，按需执行笛卡尔积，统一用df.query筛选
    返回: (error_set, cover_set) -> 集合元素为 (row_idx, col_name)
    """
    premise = rule_dict['premise']
    consequence = rule_dict['consequence']

    # 核心步骤：提取结论谓词的列名
    try:
        cons_col = extract_consequence_col(consequence)
    except ValueError:
        print(f"[WARN] Skipping rule with complex consequence: {consequence}")
        return set(), set()

    # 步骤1：预处理规则
    premise_no_quote = remove_unary_quote(premise, df_test)
    conseq_no_quote = remove_unary_quote(consequence, df_test)
    
    # 步骤2：判断谓词类型
    all_unary = is_all_unary_predicate(premise)

    if all_unary:
        def preprocess_unary(expr):
            return expr.replace('t1.', '')

        premise_final = preprocess_unary(premise_no_quote)
        conseq_final = preprocess_unary(conseq_no_quote)

        # cover
        try:
            cover_df = df_test.query(premise_final)
            cover_rows = set(cover_df.index.tolist())
        except Exception as e:
            return set(), set()

        # error
        filter_cond = f"{premise_final} and not ({conseq_final})"
        try:
            error_df = df_test.query(filter_cond)
            error_rows = set(error_df.index.tolist())
        except Exception as e:
            return set(), set()

        cover_set = {(row, cons_col) for row in cover_rows}
        error_set = {(row, cons_col) for row in error_rows}
        return error_set, cover_set

    else:
        def preprocess_binary(expr):
            return expr.replace('t1.', 't1_').replace('t2.', 't2_')

        premise_sql = preprocess_binary(premise_no_quote)
        conseq_sql = preprocess_binary(conseq_no_quote)

        # 笛卡尔积
        df_t1 = df_test.add_prefix('t1_').reset_index(names='t1_idx')
        df_t2 = df_train.add_prefix('t2_').reset_index(names='t2_idx')
        cartesian_df = pd.merge(df_t1, df_t2, how='cross')

        # cover
        try:
            cover_df = cartesian_df.query(premise_sql)
            cover_rows = set(cover_df['t1_idx'].unique().tolist())
        except Exception as e:
            return set(), set()

        # error
        filter_sql = f"{premise_sql} and not ({conseq_sql})"
        try:
            error_df = cartesian_df.query(filter_sql)
            error_rows = set(error_df['t1_idx'].unique().tolist())
        except Exception as e:
            return set(), set()

        cover_set = {(row, cons_col) for row in cover_rows}
        error_set = {(row, cons_col) for row in error_rows}
        return error_set, cover_set

def calculate_f1_score(df_dirty, df_clean, df_train, rules):
    """
    Calculates Precision, Recall, F1 for Error Detection (AND semantics).
    """
    print("[EVAL] Starting Error Detection Evaluation...")

    # 1. Ground Truth Errors
    gt_errors = set()
    eval_cols = DCRConfig.TARGET_COLS

    for col in eval_cols:
        if col not in df_dirty.columns:
            continue
        # Handle simple string comparison for diff
        diff_mask = (df_dirty[col].astype(str) != df_clean[col].astype(str))
        diff_indices = np.where(diff_mask)[0]

        for idx in diff_indices:
            gt_errors.add((idx, col))

    print(f"[EVAL] Ground Truth Errors in Targets: {len(gt_errors)}")
    if len(gt_errors) == 0:
        return 0, 0, 0, []

    # 2. Apply Rules (AND semantics)
    cover_map = defaultdict(set)   # (row,col) -> set(rule_id)
    error_map = defaultdict(set)   # (row,col) -> set(rule_id)
    rule_stats = []

    # 使用 tqdm 显示进度
    for rid, rule in enumerate(tqdm(rules, desc="Applying Rules")):
        error_set, cover_set = find_rule_error_pairs(df_dirty, rule, df_train)

        for x in cover_set:
            cover_map[x].add(rid)
        for x in error_set:
            error_map[x].add(rid)

        # Rule-level stats for logging
        if len(error_set) == 0:
            rule_stats.append({
                "rule_id": rid, "premise": rule["premise"], "consequence": rule["consequence"],
                "pred_cnt": 0, "tp_cnt": 0, "fp_cnt": 0, "rule_precision": 0.0
            })
            continue

        tp_cnt = len(error_set & gt_errors)
        fp_cnt = len(error_set - gt_errors)
        pred_cnt = tp_cnt + fp_cnt

        rule_stats.append({
            "rule_id": rid,
            "premise": rule["premise"],
            "consequence": rule["consequence"],
            "pred_cnt": pred_cnt,
            "tp_cnt": tp_cnt,
            "fp_cnt": fp_cnt,
            "rule_precision": tp_cnt / pred_cnt if pred_cnt > 0 else 0.0,
            "rule_fp_rate": fp_cnt / pred_cnt if pred_cnt > 0 else 0.0
        })

    # 3. AND 判错聚合
    predicted_errors = set()
    for x in cover_map:
        covered_rules = cover_map[x]
        error_rules = error_map.get(x, set())
        # AND 语义：被覆盖 且 所有覆盖它的规则都判错
        if len(covered_rules) > 0 and covered_rules == error_rules:
            predicted_errors.add(x)

    print(f"[EVAL] Predicted Errors (AND): {len(predicted_errors)}")

    # 4. Metrics
    tp = len(predicted_errors & gt_errors)
    fp = len(predicted_errors - gt_errors)
    fn = len(gt_errors - predicted_errors)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, rule_stats

# =========================================================
# 8. Main Execution
# =========================================================
def generate_action_space(df):
    """
    Generates Tuple-Aware Action Space with Noise Filtering.
    [优化点]
    1. 黑名单过滤：直接屏蔽 background_*, bleed 等无关列。
    2. 频次过滤：剔除 'NORWAY' 这种长尾罕见值，防止过拟合。
    """
    actions = []
    
    # =========================================================
    # 1. 定义列过滤策略 (Column Filtering)
    # =========================================================
    
    # [Hard Filter] 明确的黑名单：这些列不参与规则生成
    # 根据你的json文件分析，这些列产生了大量废话
    IGNORE_COLS = [
        "img_id", "patient_id", "lesion_id"
    ]

    BANNED_FEATURES = DCRConfig.TARGET_COLS  # ["diagnostic"]
    
    # [Soft Filter] 最小频次阈值
    # 一个值至少要在 3% 的数据中出现，才配做成规则。
    # 比如 1000 条数据，至少要有 30 条。杀掉 'NORWAY' (3条) 这种规则。
    valid_cols = [c for c in df.columns if c not in IGNORE_COLS and c not in BANNED_FEATURES]
    
    min_count = len(df) * 0.03
    print(f"[Action Space] Filtering columns. Ignored: {IGNORE_COLS}")
    print(f"[Action Space] Min value count threshold: {min_count:.1f}")

    num_cols = [
        c for c in valid_cols 
        if pd.api.types.is_numeric_dtype(df[c]) 
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    cat_cols = [
        c for c in valid_cols 
        if c not in num_cols 
        and df[c].nunique() < 30 
    ]

    print(f"[Action Space] Filtering columns. Ignored: {IGNORE_COLS}")
    print(f"[Action Space] Min value count threshold: {min_count:.1f}")

    # =========================================================
    # 2. 生成 Categorical 谓词 (带频次过滤)
    # =========================================================
    for col in cat_cols:
        value_counts = df[col].value_counts()
        common_vals = value_counts[value_counts >= min_count].index.tolist()
        
        if len(common_vals) < len(value_counts):
            print(f"   - Col '{col}': Dropped {len(value_counts) - len(common_vals)} rare values.")

        for v in common_vals:
            # Unary: t1.region == 'ARM'
            actions.append(Predicate('unary', col, '==', val=v, alias='t1'))
            actions.append(Predicate('unary', col, '==', val=v, alias='t2'))
    
    for col in cat_cols:
        # Binary: t1.region == t2.region (逻辑比较通常保留)
        actions.append(Predicate('binary', col, '==', other_col=col))

    # =========================================================
    # 3. 生成 Numerical 谓词 (分箱处理)
    # =========================================================
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0: continue
        
        # 使用 3 分位数 (25%, 50%, 75%)，保证切分点有物理意义
        quantiles = [0.5]
        bins = series.quantile(quantiles).unique()
        
        for v in bins:
            # Unary
            actions.append(Predicate('unary', col, '>', val=v, alias='t1'))
            actions.append(Predicate('unary', col, '<=', val=v, alias='t1'))
            actions.append(Predicate('unary', col, '>', val=v, alias='t2'))
            actions.append(Predicate('unary', col, '<=', val=v, alias='t2'))
            
    for col in num_cols:
        # Binary
        actions.append(Predicate('binary', col, '==', other_col=col))

    return actions

def get_df_hash(df):
    # 将 dataframe 转换为 json 字符串计算 hash，确保内容一致
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def main():
    setup_seed(42)
    os.makedirs(DCRConfig.OUT_DIR, exist_ok=True)
    noise_levels = [0, 5, 10, 15, 20]

    print(f"[INFO] Initializing Process Pool with {min(cpu_count(), 32)} workers...")
    with Pool(processes=min(cpu_count(), 32), maxtasksperchild=10) as pool:    
        for noise_val in noise_levels:
            noise_tag = f"noise_{noise_val}"
            print(f"\n{'#'*60}\n[DCR] Processing MU Noise Level: {noise_val} ({noise_tag})\n{'#'*60}")
            
            # 1. Load Training Data (Fixed Clean MR)
            # 增加打印以确认路径
            print(f"[INFO] Loading Fixed MR: {DCRConfig.FIXED_CLEAN_MR_PATH}")
            if not os.path.exists(DCRConfig.FIXED_CLEAN_MR_PATH):
                print(f"[ERR] Fixed MR file not found!")
                continue
            df_mr = pd.read_csv(DCRConfig.FIXED_CLEAN_MR_PATH)

            # B. Load MU Noise File
            if noise_val == 0:
                # [FIX] 修复路径拼接逻辑，直接使用完整路径
                mu_path = os.path.join(DCRConfig.DATA_DIR, "outputs", "mu_pred.jsonl") 
            else:
                mu_filename = f"mu_pred_noise_{noise_val}.jsonl"
                mu_path = os.path.join(DCRConfig.MU_NOISE_DIR, mu_filename)
            
            if os.path.exists(mu_path):
                print(f"[INFO] Loading MU File: {mu_path}")
                df_mu = pd.read_json(mu_path, lines=True)
                
                # [TRAINING PHASE] Keep 'None' as string for training
                # df_mu.replace(["none", "None", "NONE"], np.nan, inplace=True)
                
                df = df_mr.merge(df_mu, on="img_id", how="left")
                print(f"[DEBUG] Input DataFrame Shape: {df.shape}")
                print(f"[DEBUG] Input DataFrame Hash:  {get_df_hash(df)}")
            else:
                print(f"[WARN] MU file not found: {mu_path}. Skipping.")
                continue
            
            # 2. Setup Agent
            all_actions = generate_action_space(df)
            action_map = {a: i for i, a in enumerate(all_actions)}
            action_size = len(all_actions)
            print(f"[INFO] Action Space: {action_size} predicates")
            
            global_agent = DCRNeuralAgent(action_size, DCRConfig.HIDDEN_DIM)
            optimizer = optim.Adam(global_agent.parameters(), lr=DCRConfig.LEARNING_RATE)
            memory = deque(maxlen=DCRConfig.MEMORY_SIZE)
            
            p0_candidates = []
            for col in DCRConfig.TARGET_COLS:
                if col in df.columns:
                    vals = df[col].dropna().unique()
                    for v in vals:
                        p0_candidates.append(Predicate('unary', col, '==', val=v, alias='t1'))
            
            best_rules_global = {}

            # 3. Training Loop
            for outer_iter in range(DCRConfig.OUTER_ITERATIONS):
                print(f"\n--- Outer Iteration {outer_iter+1}/{DCRConfig.OUTER_ITERATIONS} ---")
                
                agent_state = global_agent.state_dict()
                tasks = []
                for i, p0 in enumerate(p0_candidates):
                    global_task_id = outer_iter * 10000 + i 
                    tasks.append((global_task_id, df, p0, all_actions, action_map, agent_state))
                
                results = list(tqdm(pool.imap(worker_mcts_search, tasks), 
                                    total=len(tasks), desc="MCTS Search"))
                
                new_trajectories = []
                for rules, trajs in results:
                    new_trajectories.extend(trajs)
                    for r in rules:
                        k = f"{r['premise']} -> {r['consequence']}"
                        if k not in best_rules_global:
                            best_rules_global[k] = r
                        elif r['confidence'] > best_rules_global[k]['confidence']:
                            best_rules_global[k] = r
                
                memory.extend(new_trajectories)
                loss = train_agent(global_agent, list(memory), optimizer, action_size)
                print(f"[INFO] Training Loss: {loss:.4f}")
            
            # 4. Save Results
            final_rules = list(best_rules_global.values())
            final_rules.sort(key=lambda x: x['confidence'], reverse=True)
            
            out_file = os.path.join(DCRConfig.OUT_DIR, f"dcr_rules_{noise_tag}.json")
            with open(out_file, 'w') as f:
                json.dump(final_rules, f, indent=2)
            
            print(f"[SUCCESS] Discovered {len(final_rules)} rules. Saved to {out_file}")
            if final_rules:
                print(f"Top Rule: {final_rules[0]['premise']} -> {final_rules[0]['consequence']} (Conf: {final_rules[0]['confidence']:.2f})")

            # =========================================================
            # 5. Error Detection Evaluation 
            # =========================================================
            print(f"\n--- Evaluating on Test Data ({noise_tag}) ---")
            
            filter_cols = []
            original_count = len(final_rules)
            final_rules = [
                r for r in final_rules 
                if not any(col in r['premise'] or col in r['consequence'] for col in filter_cols)
            ]
            print(f"[FILTER] Removed {original_count - len(final_rules)} rules containing blocked columns.")
            print(f"[FILTER] Final Rule Count for Eval: {len(final_rules)}")

            # Load Test Data
            dirty_path = os.path.join(DCRConfig.TEST_MR_DIR, "mr_metadata_test_dirty_augmented.csv")
            clean_path = os.path.join(DCRConfig.TEST_MR_DIR, "mr_metadata_test_clean_augmented.csv")
            
            if not os.path.exists(dirty_path): continue
                
            df_test_dirty = pd.read_csv(dirty_path)
            df_test_clean = pd.read_csv(clean_path)

            if os.path.exists(mu_path):
                df_mu_eval = pd.read_json(mu_path, lines=True)
                
                df_test_dirty = df_test_dirty.merge(df_mu_eval, on="img_id", how="left")
                df_test_clean = df_test_clean.merge(df_mu_eval, on="img_id", how="left")
            
            df_test_dirty = df_test_dirty.sort_values("img_id").reset_index(drop=True)
            df_test_clean = df_test_clean.sort_values("img_id").reset_index(drop=True)

            p, r, f1, rule_stats = calculate_f1_score(df_test_dirty, df_test_clean, df, final_rules)
            
            print(f"\n{'='*40}")
            print(f"RESULTS FOR {noise_tag}")
            print(f"Precision : {p:.4f}")
            print(f"Recall    : {r:.4f}")
            print(f"F1 Score  : {f1:.4f}")
            print(f"{'='*40}\n")
            
            metrics = {"noise": noise_tag, "precision": p, "recall": r, "f1": f1}
            with open(os.path.join(DCRConfig.OUT_DIR, f"metrics_{noise_tag}.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
            # Save Rule Stats
            rule_stats = sorted(rule_stats, key=lambda x: x.get("fp_cnt", 0.0), reverse=True)
            stat_path = os.path.join(DCRConfig.OUT_DIR, f"rule_stats_{noise_val}.json")
            with open(stat_path, "w") as f:
                json.dump(rule_stats, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Rule FP stats saved to {stat_path}")


if __name__ == "__main__":
    main()