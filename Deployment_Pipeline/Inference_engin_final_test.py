# Generated from: MAS_Testing.ipynb
# Converted at: 2026-05-17T05:35:09.839Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## MAS Testing Notebook â€” Shared Blackboard Multi-Agent Insider Threat Detection
# 
# **Inference-only pipeline (no training). All models loaded from saved files.**
# 
# - Agent 1 (CAE): `agent1_cae_new.pth` + `scaler.pkl` + `cae_threshold.npy`
# - Agent 2 (DistilBERT): MPNet snippet extraction + fine-tuned DistilBERT (same strategy as `testing_175.ipynb`)
# - Agent 3 (Temporal): `agent3_baselines_new.pkl`
# - Agent 4 (IF Orchestrator): `agent4_orchestrator_new.pkl` + `if_threshold.npy`
# - Data: `Test_sessions.csv`


import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries Imported")

# ### Configuration


# =========================================================
# PATHS â€” update these to match your file locations
# =========================================================
CONFIG = {
    # --- Test Data ---
    "test_csv": "/home/server/fyp_pipeline/Test_sessions.csv",

    # --- Saved Model Files ---
    "agent1_path":        "/home/server/fyp_pipeline/SavedModels/agent1_cae_v10ds.pth",
    "scaler_path":        "/home/server/fyp_pipeline/SavedModels/scaler.pkl",
    "cae_threshold_path": "/home/server/fyp_pipeline/SavedModels/cae_threshold.npy",
    "agent3_path":        "/home/server/fyp_pipeline/SavedModels/agent3_baselines_v10ds.pkl",
    "agent4_path":        "/home/server/fyp_pipeline/SavedModels/agent4_orchestrator_v10ds.pkl",
    "if_threshold_path":  "/home/server/fyp_pipeline/SavedModels/if_threshold_100.npy",

    # --- Agent 2 (DistilBERT) ---
    "agent2_model_dir": "/home/server/fyp_pipeline/SavedModels/final_insider_threat_model",
    "agent2_scanner":   "all-mpnet-base-v2",
}



# --- Blackboard hyper-parameters (must match training) ---
NLP_BOOST_MULTIPLIER = 1.5   # boost factor when CAE flags a session
NLP_FLAG_PERCENTILE  = 90    # percentile above which NLP score is "high"

# --- Test set composition ---
# ALL malicious sessions are always kept.
# Benign sessions are capped at this number (None = keep all).
DOWNSAMPLE_TEST_BENIGN = 10_000

# --- Numerical feature columns (must match training order exactly) ---
NUM_COLS = [
    'duration', 'is_weekend', 'is_after_hour', 'emails_count',
    'ext_emails_count', 'attachments_count', 'total_email_size',
    'http_count', 'cloud_uploads_count', 'usb_connects_count',
    'file_copies_count', 'file_to_usb_count'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("Configuration loaded.")

# ### Shared Blackboard


class Blackboard:
    """
    Shared communication hub for the multi-agent system.
    Agents write their scores, thresholds, and flags here.
    Downstream agents read upstream results to adapt their behaviour.
    """
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.data = {}
        self.messages = []

    def write(self, agent_name, key, value, msg=None):
        self.data[f"{agent_name}:{key}"] = value
        if msg:
            self.messages.append({'agent': agent_name, 'key': key, 'msg': msg})
            print(f"  [Blackboard] {msg}")

    def read(self, agent_name, key):
        return self.data.get(f"{agent_name}:{key}", None)

    def get_flag(self, agent_name, key):
        arr = self.read(agent_name, key)
        if arr is None:
            return np.zeros(self.n_samples, dtype=bool)
        return arr.astype(bool)

    def get_messages(self):
        return list(self.messages)

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  Blackboard Summary ({len(self.data)} entries, {len(self.messages)} messages)")
        print(f"{'='*60}")
        for key, val in self.data.items():
            if isinstance(val, np.ndarray):
                print(f"  {key}: ndarray shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {type(val).__name__} = {val}")
        for m in self.messages:
            print(f"  [{m['agent']}] {m['msg']}")
        print(f"{'='*60}\n")

print("Blackboard class defined.")

# ### Model Class Definitions


# =========================================================
# Agent 1 â€” Contractive Autoencoder
# =========================================================
class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ContractiveAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, input_dim)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1   = self.relu(self.fc1(x))
        h2   = self.sigmoid(self.fc2(h1))   # latent code
        h3   = self.relu(self.fc3(h2))
        recon = self.sigmoid(self.fc4(h3))
        return recon, h2


# =========================================================
# Agent 2 â€” MPNet Scanner + Augmented DistilBERT (Binary)
# =========================================================
class DistilBertThreatAgent:
    """
    Two-stage pipeline (Augmentation-250 strategy):
      Stage 1 (MPNet): Chunk text (size=1200, overlap=200) â†’ cosine similarity
                       against a single combined malicious anchor â†’ pick the
                       most suspicious chunk per session.
      Stage 2 (DistilBERT): Binary classifier (Normal=0, Malicious=1).
      Score = P(Malicious) = probs[:, 1]  (higher = more threatening).
    """

    # Single combined anchor covering both job-search and exfiltration signals
    MALICIOUS_ANCHOR_TEXT = (
        "I want to quit my job. I am applying for a new position. "
        "looking for a new job, applying for positions, sending resume, interview, recruiter, linkedin, "
        "glassdoor, monster.com, taleo, careerbuilder, layoff rumors, downsizing anxiety, severance package, "
        "Boeing, Lockheed Martin, Raytheon, Northrop Grumman, mushrooms, suillus spraguei, bad date email signature, "
        "updating cv, reference check, salary negotiation, I am stealing company secrets. "
        "I am uploading confidential files to a personal account. "
        "stealing proprietary data, uploading confidential files to cloud storage, dropbox, google drive, "
        "wikileaks, julian assange, the real story about dtaa, jainism, anekantavada, gandhi, "
        "m249 saw, fuzes, hazard analysis, strategic systems program, hms lion, uranus operation, cockatoo, "
        "copying files to usb, removable media, thumb drive, tor browser, encryption, steganography"
    )

    LABEL_MAP = {0: "Normal", 1: "Malicious"}

    def __init__(self, scanner_model_name, classifier_dir, device):
        self.device = device
        print(f"Agent 2: Loading MPNet scanner ({scanner_model_name})...")
        self.scanner = SentenceTransformer(scanner_model_name, device=str(device))
        # Single combined anchor vector
        self.anchor_vec = self.scanner.encode(self.MALICIOUS_ANCHOR_TEXT, convert_to_tensor=True)
        print(f"Agent 2: Loading fine-tuned DistilBERT (Augmentation-250) from {classifier_dir}...")
        self.tokenizer  = AutoTokenizer.from_pretrained(classifier_dir)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_dir)
        self.classifier.to(device)
        self.classifier.eval()
        print("Agent 2: Models loaded.")

    def _prepare_full_text(self, df):
        return (
            "EMAIL: "           + df['email_content_text'].fillna('').astype(str) +
            " | HTTP URL: "     + df['http_url_text'].fillna('').astype(str) +
            " | HTTP CONTENT: " + df['http_content_text'].fillna('').astype(str) +
            " | FILE NAMES: "   + df['file_names_text'].fillna('').astype(str) +
            " | FILE CONTENT: " + df['file_content_text'].fillna('').astype(str)
        ).tolist()

    def _get_best_snippet(self, text):
        """
        Stage 1: Chunk text (size=1200, overlap=200), encode with MPNet,
        return the chunk most similar to the combined malicious anchor.
        """
        if pd.isna(text) or text.strip() == "":
            return ""
        chunk_size, overlap = 1200, 200
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        if not chunks:
            chunks = [text[:512]]
        chunks = chunks[:100]   # cap at 100 chunks per session

        embs   = self.scanner.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(embs, self.anchor_vec).squeeze()
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        return chunks[torch.argmax(scores).item()]

    def _classify_batch(self, snippets):
        """
        Stage 2: Binary DistilBERT inference.
        Returns softmax probabilities of shape (batch_size, 2).
        """
        inputs = self.tokenizer(snippets, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    def predict_score(self, df, batch_size=128):
        """
        Full two-stage prediction.
        Returns:
            threat_probs  â€” P(Malicious) per session  (used as anomaly score)
            pred_labels   â€” argmax class label (0=Normal, 1=Malicious)
        """
        print("Agent 2: Preparing text data...")
        texts            = self._prepare_full_text(df)
        all_threat_probs = np.zeros(len(texts))
        all_pred_labels  = np.zeros(len(texts), dtype=int)

        print(f"Agent 2: Running MPNet + DistilBERT on {len(texts):,} sessions...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Agent 2 (DistilBERT)"):
            batch_texts    = texts[i:i + batch_size]
            snippets       = [self._get_best_snippet(t) for t in batch_texts]
            non_empty_mask = [bool(s.strip()) for s in snippets]

            if any(non_empty_mask):
                ne_snippets = [s for s, m in zip(snippets, non_empty_mask) if m]
                ne_indices  = [j for j, m in enumerate(non_empty_mask) if m]
                probs        = self._classify_batch(ne_snippets)   # (n, 2)
                # Score = P(Malicious) = probs[:, 1]
                threat_probs = probs[:, 1]
                pred_labels  = np.argmax(probs, axis=1)
                for idx, orig in enumerate(ne_indices):
                    all_threat_probs[i + orig] = threat_probs[idx]
                    all_pred_labels[i + orig]  = pred_labels[idx]

        for cid, cname in self.LABEL_MAP.items():
            print(f"  Agent 2 â€” {cname}: {(all_pred_labels == cid).sum():,} sessions")
        return all_threat_probs, all_pred_labels

    def predict_score_with_blackboard(self, df, blackboard, boost_multiplier=1.5, batch_size=128):
        """
        Blackboard-aware scoring:
        - Computes base P(Malicious) scores.
        - Reads Agent 1's high_flag from the blackboard.
        - Boosts score by `boost_multiplier` for sessions CAE-flagged as high anomaly.
        """
        base_scores, pred_labels = self.predict_score(df, batch_size=batch_size)
        cae_flag = blackboard.get_flag('agent1', 'high_flag')

        if cae_flag is not None and cae_flag.any():
            n_boosted    = cae_flag.sum()
            boosted = np.where(cae_flag,
                               np.clip(base_scores * boost_multiplier, 0.0, 1.0),
                               base_scores)
            blackboard.write('agent2', 'scores', boosted,
                             msg=f"Agent2 boosted {n_boosted} sessions by "
                                 f"{boost_multiplier}x (Agent1 CAE flag)")
        else:
            boosted = base_scores
            blackboard.write('agent2', 'scores', base_scores,
                             msg="Agent2 ran without boost (no Agent1 flags)")

        blackboard.write('agent2', 'base_scores', base_scores)
        blackboard.write('agent2', 'pred_labels', pred_labels)
        return boosted


# =========================================================
# Agent 3 â€” Temporal Baseline Agent
# =========================================================
class TemporalBaselineAgent:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.user_means = self.user_stds = self.global_mean = self.global_std = None

    def predict_score(self, test_df):
        """Per-user Z-score deviation."""
        means_dict = self.user_means.to_dict('index')
        stds_dict  = self.user_stds.to_dict('index')
        x     = test_df[self.feature_cols].values.astype(np.float64)
        means = np.zeros(x.shape, dtype=np.float64)
        stds  = np.ones(x.shape,  dtype=np.float64)
        for i, user in enumerate(test_df['user']):
            if user in means_dict:
                means[i] = list(means_dict[user].values())
                stds[i]  = list(stds_dict[user].values())
            else:
                means[i] = self.global_mean.values
                stds[i]  = self.global_std.values
        stds = np.maximum(stds, 1e-6)
        z = np.clip((x - means) / stds, -10.0, 10.0)
        return np.sqrt(np.mean(z**2, axis=1))

    def _compute_global_z_scores(self, test_df):
        x = test_df[self.feature_cols].values.astype(np.float64)
        gm = np.tile(self.global_mean.values, (len(x), 1))
        gs = np.maximum(np.tile(self.global_std.values, (len(x), 1)), 1e-6)
        z  = np.clip((x - gm) / gs, -10.0, 10.0)
        return np.sqrt(np.mean(z**2, axis=1))

    def predict_score_with_blackboard(self, test_df, blackboard):
        print("Agent 3: Computing per-user Z-Score Deviations...")
        base_scores      = self.predict_score(test_df)
        cae_high         = blackboard.get_flag('agent1', 'high_flag')
        nlp_high         = blackboard.get_flag('agent2', 'high_flag')
        upstream_flagged = cae_high | nlp_high
        n_flagged        = upstream_flagged.sum()
        if n_flagged > 0:
            print(f"Agent 3: {n_flagged} sessions flagged upstream â€” computing global baselines...")
            global_scores   = self._compute_global_z_scores(test_df)
            enhanced_scores = np.where(upstream_flagged,
                                       np.maximum(base_scores, global_scores),
                                       base_scores)
            blackboard.write('agent3', 'scores', enhanced_scores,
                             msg=f"Agent3 enhanced {n_flagged} sessions with global baselines")
            blackboard.write('agent3', 'base_scores', base_scores)
            return enhanced_scores
        else:
            print("Agent 3: No upstream flags â€” using standard per-user scoring.")
            blackboard.write('agent3', 'scores', base_scores,
                             msg="Agent3 ran without enhancement (no upstream flags)")
            blackboard.write('agent3', 'base_scores', base_scores)
            return base_scores

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.user_means  = data['user_means']
        self.user_stds   = data['user_stds']
        self.global_mean = data['global_mean']
        self.global_std  = data['global_std']
        print(f"Agent 3: Loaded baselines for {len(self.user_means)} users.")

print("All model classes defined.")

# ### Evaluation Helper


def evaluate_single_agent(agent_name, scores, y_true):
    """Compute ROC-AUC, AP and print a report at the best threshold."""
    if len(np.unique(y_true)) < 2:
        print(f"  [{agent_name}] Cannot compute AUC â€” only one class present.")
        return float('nan'), scores

    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    print(f"\n{'='*60}")
    print(f"  {agent_name} â€” Individual Evaluation")
    print(f"{'='*60}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Average Precision (AP): {ap:.6f}")

    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, scores)
    target_recall = 0.90
    valid_mask = recalls[:-1] >= target_recall
    if valid_mask.any():
        best_idx   = np.argmax(precisions[:-1][valid_mask])
        chosen_idx = np.where(valid_mask)[0][best_idx]
    else:
        f1 = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        chosen_idx = np.argmax(f1)
    best_threshold = thresholds_pr[chosen_idx]
    print(f"  Optimal threshold (recallâ‰¥{target_recall:.0%}): {best_threshold:.6f}")
    print(f"  â†’ Precision: {precisions[chosen_idx]:.4f}, Recall: {recalls[chosen_idx]:.4f}")

    preds = (scores >= best_threshold).astype(int)
    print(f"\n  Classification Report at optimal threshold:")
    print(classification_report(y_true, preds, target_names=['Benign', 'Malicious'], zero_division=0))
    n_mal     = y_true.sum()
    n_detected = ((preds == 1) & (y_true == 1)).sum()
    n_fp       = ((preds == 1) & (y_true == 0)).sum()
    print(f"  Malicious detected: {n_detected}/{n_mal}  |  False positives: {n_fp}")
    return auc, scores

print("Evaluation helper defined.")

# ### Step 1 â€” Load Test Data (`Test_sessions.csv`)


print("Loading Test_sessions.csv...")
df_raw = pd.read_csv(CONFIG['test_csv'])

# Restore datetime columns if present
for col in ['start', 'end']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')

# Ensure text columns exist and are strings
text_cols = ['email_content_text', 'http_url_text', 'http_content_text',
             'file_names_text', 'file_content_text']
for col in text_cols:
    if col not in df_raw.columns:
        df_raw[col] = ""
    else:
        df_raw[col] = df_raw[col].fillna("").astype(str)

# â”€â”€ Label handling â€” works with OR without a label column â”€â”€â”€â”€â”€â”€
# When running from ELK pipeline: no labels â†’ inference-only mode
# When running with Test_sessions.csv: labels present â†’ full evaluation
#HAS_LABELS = 'label' in df_raw.columns
HAS_LABELS = False

# â”€â”€ TESTING MODE â€” limit to first N sessions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Set TEST_MODE = True  â†’ runs on only TEST_LIMIT sessions (fast)
# Set TEST_MODE = False â†’ runs on ALL sessions (full pipeline)
# No duplicates either way â€” deterministic doc IDs guarantee upsert
TEST_MODE  = False
TEST_LIMIT = 10
if TEST_MODE:
    df_raw = df_raw.head(TEST_LIMIT).copy()
    print(f"  [TEST MODE] Limited to first {TEST_LIMIT} sessions.")
    print(f"  â†’ To run full pipeline set TEST_MODE = False")

if HAS_LABELS:
    print("  Labels detected â€” running in EVALUATION mode.")
    df_malicious = df_raw[df_raw['label'] > 0].copy()
    df_benign    = df_raw[df_raw['label'] == 0].copy()
    if DOWNSAMPLE_TEST_BENIGN is not None and len(df_benign) > DOWNSAMPLE_TEST_BENIGN:
        df_benign = df_benign.sample(n=DOWNSAMPLE_TEST_BENIGN, random_state=42)
        print(f"  Benign downsampled to {DOWNSAMPLE_TEST_BENIGN:,}")
    test_df = pd.concat([df_malicious, df_benign]).sample(frac=1, random_state=42).reset_index(drop=True)
    y_test  = (test_df['label'] > 0).astype(int).values
    print(f"\n  Test set: {(y_test == 0).sum():,} Benign  |  {y_test.sum()} Malicious")
    print(f"  Total:    {len(test_df):,} sessions")
    if y_test.sum() > 0:
        print(f"  Imbalance ratio: 1:{(y_test == 0).sum() // y_test.sum()}")
else:
    print("  No label column found â€” running in INFERENCE-ONLY mode.")
    print("  Evaluation metrics skipped. Results will be pushed to Kibana dashboard.")
    test_df = df_raw.copy()
    y_test  = np.zeros(len(test_df), dtype=int)  # dummy â€” not used for evaluation
    print(f"\n  Total sessions: {len(test_df):,}")

test_df.head()

# ### Step 2 â€” Agent 1 (CAE): Load & Score


print("\n" + "="*60)
print("  AGENT 1 â€” Contractive Autoencoder (CAE)")
print("="*60)

# â”€â”€ Load scaler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading scaler...")
with open(CONFIG['scaler_path'], 'rb') as f:
    scaler = pickle.load(f)

# â”€â”€ Load saved CAE threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cae_threshold = float(np.load(CONFIG['cae_threshold_path']))
print(f"Loaded CAE threshold: {cae_threshold:.6f}")

# â”€â”€ Scale numerical features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_test_num = torch.FloatTensor(scaler.transform(test_df[NUM_COLS])).to(device)

# â”€â”€ Load CAE model weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"Loading CAE from {CONFIG['agent1_path']}...")
agent1_model = ContractiveAutoencoder(input_dim=len(NUM_COLS)).to(device)
agent1_model.load_state_dict(torch.load(CONFIG['agent1_path'], map_location=device))
agent1_model.eval()
print("Agent 1 CAE loaded.")

# â”€â”€ Compute reconstruction errors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
loss_fn = nn.MSELoss(reduction='none')
with torch.no_grad():
    recon_test, _ = agent1_model(X_test_num)
    reconstruction_errors = torch.mean(
        loss_fn(recon_test, X_test_num), dim=1
    ).cpu().numpy()

print(f"\n  Reconstruction error â€” min: {reconstruction_errors.min():.6f}, "      f"max: {reconstruction_errors.max():.6f}, mean: {reconstruction_errors.mean():.6f}")
if HAS_LABELS and y_test.sum() > 0:
    print(f"  Mean error (Benign):    {reconstruction_errors[y_test == 0].mean():.6f}")
    print(f"  Mean error (Malicious): {reconstruction_errors[y_test == 1].mean():.6f}")

# ### Step 3 â€” Initialize Blackboard & Agent 1 Writes


print("\n" + "="*60)
print("  INITIALIZING SHARED BLACKBOARD")
print("="*60)

bb = Blackboard(n_samples=len(test_df))

# Agent 1 writes
bb.write('agent1', 'scores', reconstruction_errors,
         msg=f"Agent1 wrote {len(reconstruction_errors)} reconstruction error scores")
bb.write('agent1', 'dynamic_threshold', cae_threshold,
         msg=f"Agent1 CAE threshold = {cae_threshold:.6f}")

cae_high_flag = reconstruction_errors > cae_threshold
n_flagged = cae_high_flag.sum()
bb.write('agent1', 'high_flag', cae_high_flag,
         msg=f"Agent1 flagged {n_flagged} sessions ({n_flagged/len(test_df)*100:.2f}%) above threshold")

# Evaluate Agent 1 individually
if HAS_LABELS:
    agent1_auc, _ = evaluate_single_agent("Agent 1 (CAE)", reconstruction_errors, y_test)
else:
    agent1_auc = float('nan')
    print("  [Agent 1] Inference-only mode â€” evaluation skipped.")

# ### Step 4 â€” Agent 2 (DistilBERT + MPNet): Load & Score


print("\nInitializing Agent 2 (MPNet + DistilBERT)...")
agent2 = DistilBertThreatAgent(
    scanner_model_name = CONFIG['agent2_scanner'],
    classifier_dir     = CONFIG['agent2_model_dir'],
    device             = device
)

# Score with blackboard boost (reads Agent 1 high_flag internally)
semantic_scores = agent2.predict_score_with_blackboard(
    test_df, bb, boost_multiplier=NLP_BOOST_MULTIPLIER
)

# Write NLP high-flag to blackboard
nlp_threshold = np.percentile(semantic_scores, NLP_FLAG_PERCENTILE)
nlp_high_flag = semantic_scores > nlp_threshold
bb.write('agent2', 'high_flag', nlp_high_flag,
         msg=f"Agent2 flagged {nlp_high_flag.sum()} sessions above "             f"P{NLP_FLAG_PERCENTILE} threshold = {nlp_threshold:.4f}")

if HAS_LABELS:
    agent2_auc, _ = evaluate_single_agent("Agent 2 (DistilBERT + Boost)", semantic_scores, y_test)
else:
    agent2_auc = float('nan')
    print("  [Agent 2] Inference-only mode â€” evaluation skipped.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ### Step 5 â€” Agent 3 (Temporal Baselines): Load & Score


print("\n" + "="*60)
print("  AGENT 3 â€” Temporal Baseline Agent")
print("="*60)

agent3 = TemporalBaselineAgent(feature_cols=NUM_COLS)
agent3.load(CONFIG['agent3_path'])

# Score with blackboard context (reads Agent1 + Agent2 flags)
temporal_scores = agent3.predict_score_with_blackboard(test_df, bb)
if HAS_LABELS:
    agent3_auc, _ = evaluate_single_agent("Agent 3 (Temporal + Blackboard Context)",
                                          temporal_scores, y_test)
else:
    agent3_auc = float('nan')
    print("  [Agent 3] Inference-only mode â€” evaluation skipped.")

# ### Step 6 â€” Build Orchestrator Features


def build_orchestrator_features(cae, nlp, temp):
    return np.column_stack([
        cae, nlp, temp,
        cae * temp,
        cae * nlp,
        temp * nlp,
        np.max([cae, temp, nlp],  axis=0),
        np.std([cae, temp, nlp],  axis=0),
    ])

def build_orchestrator_features_with_blackboard(bb):
    score_cae  = bb.read('agent1', 'scores')
    score_nlp  = bb.read('agent2', 'scores')
    score_temp = bb.read('agent3', 'scores')
    base_feats = build_orchestrator_features(score_cae, score_nlp, score_temp)

    cae_flag  = bb.get_flag('agent1', 'high_flag').astype(float)
    nlp_flag  = bb.get_flag('agent2', 'high_flag').astype(float)
    temp_flag = (score_temp > np.percentile(score_temp, 90)).astype(float)
    upstream_agreement = cae_flag + nlp_flag + temp_flag

    return np.column_stack([base_feats, cae_flag, upstream_agreement])

FEATURE_NAMES = [
    'CAE_score', 'DistilBERT_score', 'Temporal_score',
    'CAEÃ—Temporal', 'CAEÃ—DistilBERT', 'TemporalÃ—DistilBERT',
    'Max_score', 'Score_std',
    'CAE_flag', 'Upstream_agreement'
]

X_orch_test = build_orchestrator_features_with_blackboard(bb)
print(f"Orchestrator feature matrix: {X_orch_test.shape}")
print(f"Features: {FEATURE_NAMES}")

# ### Step 7 â€” Agent 4 (IF Orchestrator): Load & Score


print("\n" + "="*60)
print("  AGENT 4 â€” Isolation Forest Orchestrator")
print("="*60)

# Load pre-trained Isolation Forest
print(f"Loading Agent 4 from {CONFIG['agent4_path']}...")
with open(CONFIG['agent4_path'], 'rb') as f:
    orchestrator_if = pickle.load(f)
print("Agent 4 Orchestrator loaded.")

# Load saved IF threshold
if_threshold = float(np.load(CONFIG['if_threshold_path']))
print(f"Loaded IF threshold: {if_threshold:.6f}")

# Score test data
if_raw_scores  = orchestrator_if.score_samples(X_orch_test)
final_scores_if = -if_raw_scores   # negate: higher = more anomalous

print(f"\n  IF score range: [{final_scores_if.min():.4f}, {final_scores_if.max():.4f}]")
if HAS_LABELS and y_test.sum() > 0:
    print(f"  Mean score â€” Benign:    {final_scores_if[y_test == 0].mean():.4f}")
    print(f"  Mean score â€” Malicious: {final_scores_if[y_test == 1].mean():.4f}")

if_auc = roc_auc_score(y_test, final_scores_if) if (HAS_LABELS and len(np.unique(y_test)) > 1) else float('nan')
print(f"\n  Agent 4 ROC-AUC: {if_auc:.4f}")

# ### Step 8 â€” Linear Fusion Baseline


def normalize_rank(arr):
    return rankdata(arr) / len(arr)

norm_cae  = normalize_rank(reconstruction_errors)
norm_nlp  = normalize_rank(semantic_scores)
norm_temp = normalize_rank(temporal_scores)

w_cae, w_nlp, w_temp = 0.40, 0.30, 0.30
final_scores_linear = w_cae * norm_cae + w_nlp * norm_nlp + w_temp * norm_temp

linear_auc = roc_auc_score(y_test, final_scores_linear) if (HAS_LABELS and len(np.unique(y_test)) > 1) else float('nan')
linear_ap  = average_precision_score(y_test, final_scores_linear) if (HAS_LABELS and len(np.unique(y_test)) > 1) else float('nan')

print(f"\n{'='*60}")
print(f"  Linear Fusion Baseline")
print(f"{'='*60}")
print(f"  ROC-AUC:           {linear_auc:.4f}")
print(f"  Average Precision: {linear_ap:.6f}")

# ### Step 9 â€” Final Classification (Saved IF Threshold)


print(f"\n{'='*60}")
print(f"  FINAL CLASSIFICATION â€” Agent 4 Orchestrator + Blackboard")
print(f"{'='*60}")

# â”€â”€ Apply threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Labels available  â†’ use saved IF threshold (trained operating point)
# No labels (ELK pipeline) â†’ top 5% most anomalous sessions flagged
if HAS_LABELS:
    active_threshold = if_threshold
    print(f"  Using saved IF threshold : {active_threshold:.6f}")
else:
    active_threshold = np.percentile(final_scores_if, 95)
    print(f"  Saved IF threshold       : {if_threshold:.6f}  (not used â€” no labels)")
    print(f"  Using top-5% threshold   : {active_threshold:.6f}")
    print(f"  â†’ Flags only the most anomalous 5% of sessions")

predictions = (final_scores_if >= active_threshold).astype(int)
print(f"  Sessions flagged as threat: {predictions.sum()} / {len(test_df)} "
      f"({predictions.sum()/len(test_df)*100:.1f}%)")

if HAS_LABELS:
    n_detected = ((predictions == 1) & (y_test == 1)).sum()
    n_mal      = y_test.sum()
    n_fp       = ((predictions == 1) & (y_test == 0)).sum()
    print(f"  Malicious detected   : {n_detected}/{n_mal}")
    print(f"  False positives      : {n_fp}")
    print()
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, predictions,
                                 target_names=['Benign', 'Malicious'], zero_division=0))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, predictions))
    print(f"\nROC-AUC Score: {if_auc:.4f}")
    print(f"\n{'='*60}")
    print(f"  Threshold Sweep (for reference â€” varying recall targets)")
    print(f"{'='*60}")
    if len(np.unique(y_test)) > 1:
        precisions_pr, recalls_pr, thresholds_pr = precision_recall_curve(y_test, final_scores_if)
        for target in [1.0, 0.95, 0.90, 0.85, 0.80]:
            valid = recalls_pr[:-1] >= target
            if valid.any():
                best  = np.argmax(precisions_pr[:-1][valid])
                cidx  = np.where(valid)[0][best]
                t     = thresholds_pr[cidx]
                p_tmp = (final_scores_if >= t).astype(int)
                nd    = ((p_tmp == 1) & (y_test == 1)).sum()
                nfp   = ((p_tmp == 1) & (y_test == 0)).sum()
                print(f"  Target recall >= {target:.0%}: threshold={t:.6f}, "
                      f"precision={precisions_pr[cidx]:.4f}, recall={recalls_pr[cidx]:.4f}, "
                      f"detected={nd}/{n_mal}, FP={nfp}")
else:
    print()
    print("  Evaluation metrics skipped â€” no label column.")
    print("  Results pushed to Kibana dashboard.")

# ### Final Summary


print(f"\n{'='*60}")
print(f"  FINAL SUMMARY â€” MAS Testing (Shared Blackboard)")
print(f"{'='*60}")
print()
print(f"  --- Data ---")
if HAS_LABELS:
    print(f"  Test sessions: {len(test_df):,}  ({(y_test == 0).sum():,} benign  + {y_test.sum()} malicious)")
else:
    print(f"  Test sessions: {len(test_df):,}  (inference-only â€” no labels)")
print()
print(f"  --- Individual Agent ROC-AUC ---")
print(f"  Agent 1 (CAE)                 {agent1_auc:.4f}")
print(f"  Agent 2 (DistilBERT + Boost)  {agent2_auc:.4f}")
print(f"  Agent 3 (Temporal + Context)  {agent3_auc:.4f}")
print(f"  Agent 4 (IF + Blackboard)     {if_auc:.4f}")
print(f"  Linear Fusion Baseline        {linear_auc:.4f}")
print()
print(f"  --- Blackboard Activity ---")
print(f"  Messages exchanged:  {len(bb.get_messages())}")
print(f"  CAE threshold:       {cae_threshold:.6f}")
print(f"  IF  threshold used:  {active_threshold:.6f} "
      f"  ({'saved' if HAS_LABELS else 'top-5% percentile'})")
print(f"  Sessions CAE-flagged: {bb.get_flag('agent1','high_flag').sum()}")
print(f"  Sessions NLP-flagged: {bb.get_flag('agent2','high_flag').sum()}")
print()
print(f"  --- Final Detection (IF threshold = {if_threshold:.6f}) ---")
if HAS_LABELS:
    n_detected_final = ((predictions == 1) & (y_test == 1)).sum()
    n_fp_final       = ((predictions == 1) & (y_test == 0)).sum()
    print(f"  Malicious detected:  {n_detected_final}/{y_test.sum()} "
          f"({n_detected_final/max(y_test.sum(),1)*100:.1f}%)")
    print(f"  False positives:     {n_fp_final}")
    print(f"  False positive rate: {n_fp_final/max((y_test==0).sum(),1)*100:.2f}%")
else:
    print(f"  Sessions flagged:    {predictions.sum()} / {len(test_df)} (top 5% most anomalous)")
    print(f"  â†’ Open Kibana: Dashboards â†’ Insider Threat Detection Dashboard")
print(f"{'='*60}")

# ### Plots â€” Score Distribution


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax1 = axes[0]
if HAS_LABELS:
    sns.histplot(final_scores_if[y_test == 0], color='steelblue', label='Benign',
                 kde=True, stat="density", bins=50, ax=ax1)
    if y_test.sum() > 0:
        sns.histplot(final_scores_if[y_test == 1], color='crimson', label='Malicious',
                     kde=True, stat="density", bins=10, ax=ax1)
else:
    sns.histplot(final_scores_if, color='steelblue', label='All Sessions',
                 kde=True, stat="density", bins=50, ax=ax1)
ax1.axvline(if_threshold, color='green', linestyle='--',
            label=f"Saved threshold ({if_threshold:.4f})")
ax1.set_title("Agent 4 (IF Orchestrator) â€” Score Distribution")
ax1.set_xlabel("Anomaly Score (higher = more anomalous)")
ax1.set_ylabel("Density")
ax1.legend()

ax2 = axes[1]
if HAS_LABELS:
    sns.histplot(final_scores_linear[y_test == 0], color='steelblue', label='Benign',
                 kde=True, stat="density", bins=50, ax=ax2)
    if y_test.sum() > 0:
        sns.histplot(final_scores_linear[y_test == 1], color='crimson', label='Malicious',
                     kde=True, stat="density", bins=10, ax=ax2)
else:
    sns.histplot(final_scores_linear, color='steelblue', label='All Sessions',
                 kde=True, stat="density", bins=50, ax=ax2)
ax2.set_title("Linear Fusion Baseline â€” Score Distribution")
ax2.set_xlabel("Fused Rank Score")
ax2.set_ylabel("Density")
ax2.legend()

plt.tight_layout()
plt.show()

# ### Confusion Matrix
if HAS_LABELS:
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Malicious'],
                yticklabels=['Normal', 'Malicious'])
    plt.title("Confusion Matrix â€” Blackboard MAS")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
else:
    print("Confusion Matrix skipped â€” no labels (inference-only mode).")

# ### ROC Curve â€” All Agents


if HAS_LABELS and len(np.unique(y_test)) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, scores, auc_val in [
        ("Agent 1 (CAE)",             reconstruction_errors, agent1_auc),
        ("Agent 2 (DistilBERT+Boost)", semantic_scores,       agent2_auc),
        ("Agent 3 (Temporal+Context)", temporal_scores,       agent3_auc),
        ("Agent 4 (IF+Blackboard)",   final_scores_if,        if_auc),
        ("Linear Fusion Baseline",    final_scores_linear,    linear_auc),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")
    ax.plot([0,1],[0,1],'k--',label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison â€” Blackboard MAS Testing")
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
else:
    if not HAS_LABELS:
        print("ROC curve skipped â€” inference-only mode.")
    else:
        print("ROC curve requires both classes in y_test.")

# =============================================================
# ELASTICSEARCH â€” Push Predictions to Kibana Dashboard
# =============================================================
import hashlib
import logging
import sys
from datetime import datetime, timezone
from elasticsearch import Elasticsearch

ES_CONFIG = {
    "es_host":     "https://localhost:9200",
    "es_user":     "elastic",
    "es_password": "AG=6l4+qzo0r9+saUpgu",
    "es_ca_cert":  "/etc/elasticsearch/certs/http_ca.crt",
    "es_index":    "threat-predictions",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/server/fyp_pipeline/deployment_inference.log"),
    ],
)
log = logging.getLogger(__name__)

log.info("\n" + "="*60)
log.info("  ELASTICSEARCH â€” Pushing predictions to Kibana Dashboard")
log.info("="*60)

es = Elasticsearch(
    ES_CONFIG["es_host"],
    basic_auth=(ES_CONFIG["es_user"], ES_CONFIG["es_password"]),
    ca_certs=ES_CONFIG["es_ca_cert"],
    verify_certs=False,
)
INDEX = ES_CONFIG["es_index"]

ES_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":            {"type": "date"},
            "session_start":         {"type": "date"},
            "session_end":           {"type": "date"},
            "user":                  {"type": "keyword"},
            "risk_score":            {"type": "float"},
            "risk_score_normalized": {"type": "float"},
            "risk_label":            {"type": "keyword"},
            "is_threat":             {"type": "boolean"},
            "prediction":            {"type": "integer"},
            "true_label":            {"type": "integer"},
            "agent1_cae_score":      {"type": "float"},
            "agent1_cae_flag":       {"type": "boolean"},
            "agent2_nlp_score":      {"type": "float"},
            "agent2_nlp_flag":       {"type": "boolean"},
            "agent2_nlp_label":      {"type": "keyword"},
            "agent3_temporal_score": {"type": "float"},
            "upstream_agreement":    {"type": "integer"},
            "duration":              {"type": "float"},
            "is_weekend":            {"type": "boolean"},
            "is_after_hour":         {"type": "boolean"},
            "emails_count":          {"type": "integer"},
            "ext_emails_count":      {"type": "integer"},
            "attachments_count":     {"type": "integer"},
            "total_email_size":      {"type": "float"},
            "http_count":            {"type": "integer"},
            "cloud_uploads_count":   {"type": "integer"},
            "usb_connects_count":    {"type": "integer"},
            "file_copies_count":     {"type": "integer"},
            "file_to_usb_count":     {"type": "integer"},
        }
    }
}

if not es.indices.exists(index=INDEX):
    es.indices.create(index=INDEX, body=ES_MAPPING)
    log.info(f"[ES] Created index '{INDEX}'")
else:
    log.info(f"[ES] Index '{INDEX}' already exists â€” upsert mode (no duplicates)")

def get_risk_label(norm_score):
    if norm_score >= 0.75:   return "CRITICAL"
    elif norm_score >= 0.60: return "HIGH"
    elif norm_score >= 0.45: return "MEDIUM"
    else:                    return "LOW"

def safe_dt(val):
    try:
        return pd.to_datetime(val).isoformat()
    except Exception:
        return None

def make_doc_id(user, session_start):
    """Deterministic ID â€” same session always same ID â†’ no duplicates on re-run."""
    return hashlib.md5(f"{user}_{session_start}".encode()).hexdigest()

# Retrieve variables from blackboard
cae_flags  = bb.get_flag('agent1', 'high_flag')
nlp_flags  = bb.get_flag('agent2', 'high_flag')
nlp_labels = bb.read('agent2', 'pred_labels')
if nlp_labels is None:
    nlp_labels = np.zeros(len(test_df), dtype=int)

NLP_LABEL_MAP = {0: "Normal", 1: "Malicious"}

temp_flag_arr          = (temporal_scores > np.percentile(temporal_scores, 90)).astype(float)
upstream_agreement_arr = cae_flags.astype(float) + nlp_flags.astype(float) + temp_flag_arr

# Normalise final_scores_if to [0,1] for Kibana gauges
score_min   = final_scores_if.min()
score_max   = final_scores_if.max()
score_range = score_max - score_min if score_max > score_min else 1.0
norm_scores = (final_scores_if - score_min) / score_range

log.info(f"[ES] Indexing {len(test_df):,} sessions â†’ '{INDEX}'...")

success = created = upserted = 0

for i, (_, row) in enumerate(test_df.iterrows()):
    norm          = float(norm_scores[i])
    user          = str(row.get("user", "unknown"))
    session_start = safe_dt(row.get("start")) or str(i)
    doc_id        = make_doc_id(user, session_start)

    doc = {
        "@timestamp":            datetime.now(timezone.utc).isoformat(),
        "session_start":         session_start,
        "session_end":           safe_dt(row.get("end")),
        "user":                  user,
        "risk_score":            float(final_scores_if[i]),
        "risk_score_normalized": norm,
        "risk_label":            get_risk_label(norm),
        "is_threat":             bool(predictions[i] == 1),
        "prediction":            int(predictions[i]),
        "true_label":            int(y_test[i]),   # 0 if no labels
        "agent1_cae_score":      float(reconstruction_errors[i]),
        "agent1_cae_flag":       bool(cae_flags[i]),
        "agent2_nlp_score":      float(semantic_scores[i]),
        "agent2_nlp_flag":       bool(nlp_flags[i]),
        "agent2_nlp_label":      NLP_LABEL_MAP.get(int(nlp_labels[i]), "Unknown"),
        "agent3_temporal_score": float(temporal_scores[i]),
        "upstream_agreement":    int(upstream_agreement_arr[i]),
        "duration":              float(row.get("duration", 0)),
        "is_weekend":            bool(row.get("is_weekend", False)),
        "is_after_hour":         bool(row.get("is_after_hour", False)),
        "emails_count":          int(row.get("emails_count", 0)),
        "ext_emails_count":      int(row.get("ext_emails_count", 0)),
        "attachments_count":     int(row.get("attachments_count", 0)),
        "total_email_size":      float(row.get("total_email_size", 0)),
        "http_count":            int(row.get("http_count", 0)),
        "cloud_uploads_count":   int(row.get("cloud_uploads_count", 0)),
        "usb_connects_count":    int(row.get("usb_connects_count", 0)),
        "file_copies_count":     int(row.get("file_copies_count", 0)),
        "file_to_usb_count":     int(row.get("file_to_usb_count", 0)),
    }

    try:
        result = es.index(index=INDEX, id=doc_id, document=doc)
        success += 1
        if result.get("result") == "created":
            created  += 1
        else:
            upserted += 1
    except Exception as e:
        log.error(f"  Failed session {i} (user={user}): {e}")

log.info(f"\n{'='*60}")
log.info(f"  ES INDEXING COMPLETE")
log.info(f"  Sessions indexed  : {success}/{len(test_df)}")
log.info(f"  New docs created  : {created}")
log.info(f"  Existing updated  : {upserted}  â†� 0 duplicates on re-run")
log.info(f"  Threats flagged   : {predictions.sum()} / {len(test_df)}")
log.info(f"  ES index          : {INDEX}")
log.info(f"  â†’ Kibana: Dashboards â†’ Insider Threat Detection Dashboard")
log.info(f"  â†’ Set time range  : Last 1 hour / Today")
log.info(f"{'='*60}\n")
