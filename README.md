# DCR: Extending Dependencies from Relations to Unstructured Data

The source code of DCRs (VLDB submission) 

- MCTS + policy learning
- ML models, e.g., M_R, M_U

The following is source code for PAD dataset; for other datasets, the code is in "dcrs-others" directory.

## 🚀 Quick Start

### Step 1: Data Preprocessing

Split the dataset and generate noisy test data:

```bash
python data_preprocess.py

```
* **Input:** `metadata.csv`
* **Output:** `train_clean.csv`, `test_dirty.csv`, etc..

### Step 2: M_U model

Use **Qwen2.5-VL** to extract `border`, `surface`, and `color` attributes from images:

```bash
python mu_for_label.py  # Generate list for labeling
python mu.py            # Execute inference
python mu_test.py       # Calculate F1 Score

```
### Step 3: M_R model

Generate CLIP embeddings, perform clustering, and inject Cluster noise.

* Execute the M_R model to output the CSV files to the `mr_outputs_cluster3/` directory.
* **Key Feature:** `img_visual_cluster` (K=3).

### Step 4: DCR Rule Discovery (Core)

Run the main program **dcr_mu.py** and **dcr_mr.py** to mine rules. The program automatically executes the following pipeline:

1. **Load:** Read inference results of M_R and M_U
2. **Merge:** Fuse multimodal features via `img_id`(HER is not used in PAD).
3. **Search:** Search for predicate combinations using **MCTS + Neural Policy Network**.
4. **Eval:** Calculate Support and Confidence.

### Step 5: Error Detection Evaluation

The program automatically runs evaluation on `test_dirty.csv` after rule discovery is complete:

* **Logic:** Based on **AND Semantics** (an error is flagged only if *all* applicable rules identify it as such).
* **Output:** Precision, Recall, F1-score.
