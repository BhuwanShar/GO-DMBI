import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    recall_score, accuracy_score, precision_score, f1_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import warnings
import copy
import torch.backends.cudnn
warnings.filterwarnings("ignore")

# =========================================
# CONFIGURATION
# =========================================
class Config:

    # Reproducibility
    SEED = 42

    # Data paths
    GEOKG_PATH = "DATA/Data_process_output/glio_node_features_geokg.npy"
    GO_TERMS_PATH = "DATA/Data_process_output/glio_node_features_go_terms.npy"
    NODES_PATH = "DATA/Data_process_output/glio_nodes_with_uniprot_and_embidx.csv"
    EDGES_PATH = "DATA/Raw/Glioblastoma_microarray/glio_edges.csv"
    OUTPUT_DIR = "update1_glio_outputs"

    # Model hyperparameters
    HIDDEN_DIM = 512
    PROJ_DIM = 64
    NUM_GRAPH_LAYERS = 16
    DROPOUT = 0.2
    
    # Training hyperparameters
    LR = 0.0003
    WEIGHT_DECAY = 0.01
    EPOCHS = 500
    PATIENCE = 50
    DROP_EDGE_RATE = 0.3
    
    # Loss function
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 4.3 # forces model to concentrate on hard samples
    POS_WEIGHT = 4
    FP_WEIGHT = 0.3
    
    # Contrastive learning
    SUPCON_TEMP = 0.06
    LAMBDA_SUPCON = 0.34 #  controls strength of regularization
    
    # Latent SMOTE
    SMOTE_K = 6
    SMOTE_ALPHA = 1.9
    
    # Threshold
    USE_OPTIMAL_THRESHOLD = True
    THRESHOLD_BETA = 0.7 #  this is choseen as the parameter is optimized in 10 fold cross val
    
    # Activations
    MLP_ACTIVATION = 'relu'
    GRAPH_ACTIVATION = 'gelu'
    PROJ_ACTIVATION = 'leaky_relu'

    # Cross-validation
    N_FOLDS = 10
    CV_PATIENCE = 20

    @classmethod
    def validate(cls):
        # Validate configuration
        assert 0 < cls.DROPOUT < 1, "Dropout must be between 0 and 1"
        assert cls.HIDDEN_DIM > 0, "Hidden dimension must be positive"
        assert cls.THRESHOLD_BETA > 0, "Threshold beta must be positive"
        assert 0 <= cls.FP_WEIGHT <= 1, "FP weight should be between 0 and 1"
        print("[CONFIG] Configuration validated successfully")

    @classmethod
    def to_dict(cls):
        #Convert config to dictionary for saving
        return {
            'SEED': cls.SEED,
            'HIDDEN_DIM': cls.HIDDEN_DIM,
            'PROJ_DIM': cls.PROJ_DIM,
            'NUM_GRAPH_LAYERS': cls.NUM_GRAPH_LAYERS,
            'DROPOUT': cls.DROPOUT,
            'LR': cls.LR,
            'WEIGHT_DECAY': cls.WEIGHT_DECAY,
            'EPOCHS': cls.EPOCHS,
            'PATIENCE': cls.PATIENCE,
            'DROP_EDGE_RATE': cls.DROP_EDGE_RATE,
            'USE_FOCAL_LOSS': cls.USE_FOCAL_LOSS,
            'FOCAL_GAMMA': cls.FOCAL_GAMMA,
            'POS_WEIGHT': cls.POS_WEIGHT,
            'FP_WEIGHT': cls.FP_WEIGHT,
            'SUPCON_TEMP': cls.SUPCON_TEMP,
            'LAMBDA_SUPCON': cls.LAMBDA_SUPCON,
            'SMOTE_K': cls.SMOTE_K,
            'SMOTE_ALPHA': cls.SMOTE_ALPHA,
            'USE_OPTIMAL_THRESHOLD': cls.USE_OPTIMAL_THRESHOLD,
            'THRESHOLD_BETA': cls.THRESHOLD_BETA,
            'MLP_ACTIVATION': cls.MLP_ACTIVATION,
            'GRAPH_ACTIVATION': cls.GRAPH_ACTIVATION,
            'PROJ_ACTIVATION': cls.PROJ_ACTIVATION,
            'N_FOLDS': cls.N_FOLDS,
            'CV_PATIENCE': cls.CV_PATIENCE,
        }

# ==================================
# GRAPH  INFORMATION
# ==================================
class GraphStatisticsTracker:
    '''Track and save graph statistics at different processing stages'''

    def __init__(self):
        self.stats = []

    def add_stage(self, stage_name, n_nodes, n_edges, n_positives=None,
                  n_removed_nodes=None, removal_reason=None, additional_info=None):
        '''include statistics for a processing stage'''
        entry = {
            'stage': stage_name,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_positives': n_positives,
            'n_removed_nodes': n_removed_nodes,
            'removal_reason': removal_reason,
            'additional_info': additional_info
        }
        self.stats.append(entry)
        print(f"[GRAPH STATS] {stage_name}: nodes={n_nodes}, edges={n_edges}"
              + (f", positives={n_positives}" if n_positives is not None else "")
              + (f", removed={n_removed_nodes}" if n_removed_nodes is not None else ""))

    def save_to_csv(self, filepath):
        """Save statistics to CSV"""
        df = pd.DataFrame(self.stats)
        df.to_csv(filepath, index=False)
        print(f"[GRAPH STATS] Saved to {filepath}")

    def get_dataframe(self):
        """Return statistics as DataFrame"""
        return pd.DataFrame(self.stats)

# ============================================================================
# SETUP
# ============================================================================
def setup_environment(seed):
    # Setup reproducibility and device
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Device: {device}")
    return device


def create_output_dirs(base_dir):
    """Create output directories"""
    dirs = [f"{base_dir}/plots", f"{base_dir}/models", f"{base_dir}/metrics"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[SETUP] Output directories created: {base_dir}/")

# =================================
# DATA LOADING
# =================================
class DataLoader:
    """Handles data loading with validation"""

    def __init__(self, config):
        self.config = config

    def load_features(self):
        """Load node features"""
        geokg = np.load(self.config.GEOKG_PATH)
        go_terms = np.load(self.config.GO_TERMS_PATH)

        assert geokg.shape[0] == go_terms.shape[0], "Feature dimension mismatch"
        print(f"[DATA] Loaded features: geokg={geokg.shape}, go_terms={go_terms.shape}")
        return geokg, go_terms

    def load_nodes(self):
        """Load node metadata"""
        nodes = pd.read_csv(self.config.NODES_PATH)
        assert "SYMBOL" in nodes.columns and "is_biomarker" in nodes.columns
        print(f"[DATA] Loaded nodes: {len(nodes)} entries")
        return nodes

    def load_edges(self):
        """Load edge list"""
        edges = pd.read_csv(self.config.EDGES_PATH)
        assert "source" in edges.columns and "target" in edges.columns
        print(f"[DATA] Loaded edges: {len(edges)} entries")
        return edges

    def filter_invalid_features(self, geokg, go_terms, nodes):
        """Remove nodes with invalid features (zero norm)"""
        geokg_norm = np.linalg.norm(geokg, axis=1)
        valid_mask = geokg_norm > 1e-6

        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            print(f"[DATA] Removed {n_removed} nodes with invalid features (zero norm)")

        return (
            geokg[valid_mask],
            go_terms[valid_mask],
            nodes.loc[valid_mask].reset_index(drop=True),
            valid_mask,
            n_removed
        )

    def build_edge_index(self, nodes, edges):
        """Build edge index tensor"""
        sym2idx = {s.upper(): i for i, s in enumerate(nodes["SYMBOL"])}

        valid_edges = []
        skipped = 0

        for _, r in edges.iterrows():
            s, t = r["source"].upper(), r["target"].upper()
            if s in sym2idx and t in sym2idx:
                u, v = sym2idx[s], sym2idx[t]
                if u != v:
                    valid_edges.append([u, v])
                    valid_edges.append([v, u])
            else:
                skipped += 1

        if skipped > 0:
            print(f"[DATA] Skipped {skipped} edges (nodes not found in symbol mapping)")

        if len(valid_edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

        print(f"[DATA] Built edge index: {edge_index.shape[1] // 2} undirected edges")

        return edge_index, sym2idx

# ============================================
# ISOLATED NODE REMOVAL
# ============================================
def compute_node_degrees(edge_index, n_nodes):
    """Compute degree for each node"""
    if edge_index.numel() == 0:
        return np.zeros(n_nodes, dtype=np.int64)

    edge_index_np = edge_index.cpu().numpy()
    degrees = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(degrees, edge_index_np[0], 1)
    return degrees


def remove_isolated_nodes(X, y_np, nodes, edge_index, stats_tracker):
    """Remove nodes with degree 0 (no edges)"""
    n_nodes = len(y_np)
    degrees = compute_node_degrees(edge_index, n_nodes)
    isolated_mask = degrees == 0
    n_isolated = isolated_mask.sum()

    if n_isolated == 0:
        print("[GRAPH] No isolated nodes found")
        stats_tracker.add_stage(
            stage_name="Isolated Node Check",
            n_nodes=n_nodes,
            n_edges=edge_index.shape[1] // 2,
            n_positives=y_np.sum(),
            n_removed_nodes=0,
            removal_reason="No isolated nodes"
        )
        return X, y_np, nodes, edge_index

    print(f"[GRAPH] Found {n_isolated} isolated nodes (degree=0), removing...")

    # Get indices of non-isolated nodes
    keep_mask = ~isolated_mask
    keep_indices = np.where(keep_mask)[0]

    # Create mapping from old indices to new indices
    old_to_new = np.full(n_nodes, -1, dtype=np.int64)
    old_to_new[keep_indices] = np.arange(len(keep_indices))

    # Filter features, labels, and nodes DataFrame
    X_filtered = X[keep_mask]
    y_filtered = y_np[keep_mask]
    nodes_filtered = nodes.loc[keep_mask].reset_index(drop=True)

    # Remap edge indices
    edge_index_np = edge_index.cpu().numpy()
    new_sources = old_to_new[edge_index_np[0]]
    new_targets = old_to_new[edge_index_np[1]]

    # Filter out any edges that reference removed nodes (should be none)
    valid_edge_mask = (new_sources >= 0) & (new_targets >= 0)
    new_edge_index = torch.tensor(
        np.stack([new_sources[valid_edge_mask], new_targets[valid_edge_mask]]),
        dtype=torch.long
    )

    # Log removed nodes
    removed_symbols = nodes.loc[isolated_mask, 'SYMBOL'].tolist()
    removed_biomarkers = nodes.loc[isolated_mask, 'is_biomarker'].sum()

    print(f"[GRAPH] Removed {n_isolated} isolated nodes ({removed_biomarkers} were biomarkers)")

    stats_tracker.add_stage(
        stage_name="Isolated Node Removal",
        n_nodes=len(y_filtered),
        n_edges=new_edge_index.shape[1] // 2,
        n_positives=y_filtered.sum(),
        n_removed_nodes=n_isolated,
        removal_reason="Degree = 0",
        additional_info=f"Removed biomarkers: {removed_biomarkers}"
    )

    return X_filtered, y_filtered, nodes_filtered, new_edge_index

# ====================================
# DATA SPLITTING
# ====================================
class DataSplitter:

    def __init__(self, n_nodes, y, edge_index, seed):
        self.n_nodes = n_nodes
        self.y = y
        self.full_edge_index = edge_index
        self.seed = seed

    def train_test_split(self, test_size=0.2):
        # Split into train_val and test sets
        all_idx = np.arange(self.n_nodes)

        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, stratify=self.y, random_state=self.seed
        )

        assert len(set(train_val_idx) & set(test_idx)) == 0, "DATA LEAK: Overlap in node indices"

        print(f"[SPLIT] Train/Val: {len(train_val_idx)}, Test: {len(test_idx)}")
        print(f"[SPLIT] Train/Val positives: {self.y[train_val_idx].sum()}")
        print(f"[SPLIT] Test positives: {self.y[test_idx].sum()}")

        return train_val_idx, test_idx

    def validate_no_label_leakage(self, train_idx, test_idx):
        # Verify no node index overlap
        assert len(set(train_idx) & set(test_idx)) == 0, "LABEL LEAK: Node overlap"
        print("[SPLIT] Verified: No label leakage (disjoint node sets)")

    def count_edges_for_split(self, node_indices):
        # Count edges where both endpoints are in the given node set
        edge_index_np = self.full_edge_index.cpu().numpy()
        node_set = set(node_indices)
        count = 0
        for i in range(edge_index_np.shape[1]):
            if edge_index_np[0, i] in node_set and edge_index_np[1, i] in node_set:
                count += 1
        return count // 2  # Undirected edges counted twice

# ==========================
# FEATURE NORMALIZATION 
# ===========================
class LeakFreeNormalizer:
    # Leak-free feature normalization used on trasining data only

    @staticmethod
    def normalize_for_split(X_raw, train_idx, other_idx):
        """Normalize: fit on train, transform both"""
        scaler = StandardScaler()
        X_normalized = np.zeros_like(X_raw)
        X_normalized[train_idx] = scaler.fit_transform(X_raw[train_idx])
        X_normalized[other_idx] = scaler.transform(X_raw[other_idx])
        return X_normalized, scaler

# ===============================
# UTILITY FUNCTIONS
# ===============================
def get_activation(name):
    """Return activation function by name"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'silu': nn.SiLU()
    }
    return activations.get(name.lower(), nn.ReLU())

def drop_edge(edge_index, p=0.1, training=True):
    """Randomly drop edges for regularization"""
    if not training or p == 0:
        return edge_index
    mask = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, mask]

def latent_smote(z, y, k=5, alpha=0.5):
    """SMOTE in latent space"""
    pos_indices = (y == 1).nonzero(as_tuple=True)[0]

    if len(pos_indices) < k + 1:
        return z, y

    pos_emb = z[pos_indices]
    dist = torch.cdist(pos_emb, pos_emb)
    dist.fill_diagonal_(float('inf'))
    _, indices = dist.topk(k, largest=False)

    rand_neighbor_idx = indices[:, torch.randint(0, k, (1,), device=z.device).item()]
    neighbor_emb = pos_emb[rand_neighbor_idx]

    lam = torch.rand(len(pos_indices), 1, device=z.device) * alpha
    synthetic_emb = pos_emb + lam * (neighbor_emb - pos_emb)

    z_new = torch.cat([z, synthetic_emb], dim=0)
    y_new = torch.cat([y, torch.ones(len(pos_indices), device=y.device)], dim=0)

    return z_new, y_new

def find_optimal_threshold(y_true, y_scores, beta=1.0):
    '''Find threshold maximizing F-beta'''
    best_threshold, best_score = 0.5, 0

    for t in np.linspace(0.2, 0.8, 60):
        pred = (y_scores > t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)

        if p + r > 0:
            f_beta = (1 + beta ** 2) * (p * r) / ((beta ** 2 * p) + r + 1e-8)
            if f_beta > best_score:
                best_score = f_beta
                best_threshold = t

    return best_threshold

# =========================
# LOSS FUNCTIONS
# =========================
class FocalLoss(nn.Module):
    # Focal Loss with FP penalty

    def __init__(self, gamma=2.0, fp_weight=0.0):
        super().__init__()
        self.gamma = gamma
        self.fp_weight = fp_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.fp_weight > 0:
            probs = torch.sigmoid(logits)
            fp_penalty = probs * (1 - targets)
            focal_loss = focal_loss + self.fp_weight * fp_penalty

        return focal_loss.mean()

class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temp

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(mask.shape[0], device=mask.device).view(-1, 1), 0
        )
        mask = mask * logits_mask

        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        return -mean_log_prob_pos.mean()

# =====================================
# MODEL
# ======================================
from torch_geometric.nn import GCNConv

class GCNLayer(nn.Module):
    # Symmetric GCN Layer using Kipf & Welling normalization 

    def __init__(self, in_feat, out_feat, dropout=0.3, activation='relu'):
        super().__init__()
        self.conv = GCNConv(
            in_channels=in_feat,
            out_channels=out_feat,
            add_self_loops=True,
            normalize=True
        )
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            out = self.conv.lin(x)
        else:
            out = self.conv(x, edge_index)

        out = self.activation(out)
        out = self.dropout(out)
        return out


class BiomarkerGNN(nn.Module):
    # Proposed model archetecture
    def __init__(self, n_feat, hidden_dim, num_layers, dropout, config):
        super().__init__()
        self.config = config

        # MLP branch
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            get_activation(config.MLP_ACTIVATION),
            nn.Dropout(dropout)
        )

        # GCN branch
        self.graph_layers = nn.ModuleList()
        self.graph_layers.append(
            GCNLayer(n_feat, hidden_dim, dropout, config.GRAPH_ACTIVATION)
        )
        for _ in range(num_layers - 1):
            self.graph_layers.append(
                GCNLayer(hidden_dim, hidden_dim, dropout, config.GRAPH_ACTIVATION)
            )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(config.PROJ_ACTIVATION),
            nn.Linear(hidden_dim, config.PROJ_DIM)
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, labels=None, smote=False):
        h_mlp = self.feat_mlp(x)

        h_graph = x
        for layer in self.graph_layers:
            h_graph = layer(h_graph, edge_index)

        combined = torch.cat([h_mlp, h_graph], dim=1)
        gate = self.gate(combined)
        h_fused = gate * h_graph + (1 - gate) * h_mlp

        if smote and labels is not None and self.training:
            h_fused, labels = latent_smote(
                h_fused, labels, k=self.config.SMOTE_K, alpha=self.config.SMOTE_ALPHA
            )

        z_proj = self.projection(h_fused)
        logits = self.classifier(h_fused).view(-1)

        return logits, labels, z_proj


# ============================================
# METRICS
# ============================================

def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate evaluation metrics"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
        'PR-AUC': average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

# ==========================================
# TRAINER
# ==========================================
class Trainer:
    """Model training handler"""

    def __init__(self, config, device):
        self.config = config
        self.device = device

    def create_model(self, n_features):
        """Create model"""
        return BiomarkerGNN(
            n_feat=n_features,
            hidden_dim=self.config.HIDDEN_DIM,
            num_layers=self.config.NUM_GRAPH_LAYERS,
            dropout=self.config.DROPOUT,
            config=self.config
        ).to(self.device)

    def create_criterion(self):
        """Create loss functions"""
        if self.config.USE_FOCAL_LOSS:
            cls_loss = FocalLoss(gamma=self.config.FOCAL_GAMMA, fp_weight=self.config.FP_WEIGHT)
        else:
            cls_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.config.POS_WEIGHT], device=self.device)
            )

        supcon_loss = SupervisedContrastiveLoss(temperature=self.config.SUPCON_TEMP)
        return cls_loss, supcon_loss

    def train_epoch(self, model, optimizer, X, y, edge_index, train_mask,
                    cls_criterion, supcon_criterion, N):
        """Single training epoch"""
        model.train()
        optimizer.zero_grad()

        batch_edge_index = drop_edge(edge_index, p=self.config.DROP_EDGE_RATE, training=True)
        logits, targets, z_proj = model(X, batch_edge_index, y, smote=True)

        num_syn = logits.shape[0] - N
        train_indices = torch.nonzero(train_mask).squeeze(-1)

        if num_syn > 0:
            synthetic_indices = torch.arange(N, N + num_syn, device=self.device)
            active_indices = torch.cat([train_indices, synthetic_indices])
        else:
            active_indices = train_indices

        loss = cls_criterion(logits[active_indices], targets[active_indices])
        loss += self.config.LAMBDA_SUPCON * supcon_criterion(
            z_proj[active_indices], targets[active_indices]
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    def validate(self, model, X, y, edge_index, val_mask):
        """Validation step"""
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X, edge_index, None, smote=False)
            probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
            true = y[val_mask].cpu().numpy()

            try:
                auc = roc_auc_score(true, probs)
            except ValueError:
                auc = 0.5

        return auc, probs, true

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(X_raw, y_np, full_edge_index, train_val_idx, config, device):
    """Run K-fold CV - uses FULL graph structure"""
    print("\n" + "=" * 60)
    print("10-FOLD CROSS-VALIDATION (LABEL-ONLY SPLIT)")
    print("=" * 60)
    print("[CV] Full graph structure is used (standard GNN practice)")

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    trainer = Trainer(config, device)

    cv_results = []
    cv_thresholds = []
    N = len(y_np)

    full_edge_index_device = full_edge_index.to(device)

    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_val_idx, y_np[train_val_idx])):
        print(f"\n{'=' * 25} Fold {fold + 1}/{config.N_FOLDS} {'=' * 25}")

        global_train_idx = train_val_idx[train_fold_idx]
        global_val_idx = train_val_idx[val_fold_idx]

        # Leak-free normalization (fit on train only)
        X_normalized, _ = LeakFreeNormalizer.normalize_for_split(
            X_raw, global_train_idx, global_val_idx
        )
        X_feat = torch.tensor(X_normalized, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.float32, device=device)

        print(f"  Training with FULL graph: {full_edge_index_device.shape[1] // 2} edges")

        # Masks
        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        val_mask = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[global_train_idx] = True
        val_mask[global_val_idx] = True

        # Model
        model = trainer.create_model(X_feat.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        cls_criterion, supcon_criterion = trainer.create_criterion()

        best_auc, patience_cnt, best_state = 0, 0, None

        for epoch in range(config.EPOCHS):
            loss = trainer.train_epoch(
                model, optimizer, X_feat, y, full_edge_index_device,
                train_mask, cls_criterion, supcon_criterion, N
            )
            val_auc, _, _ = trainer.validate(model, X_feat, y, full_edge_index_device, val_mask)

            if val_auc > best_auc:
                best_auc = val_auc
                patience_cnt = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_cnt += 1
                if patience_cnt >= config.CV_PATIENCE:
                    break

        # Final eval
        model.load_state_dict(best_state)
        _, val_probs, val_true = trainer.validate(model, X_feat, y, full_edge_index_device, val_mask)

        fold_threshold = find_optimal_threshold(val_true, val_probs, config.THRESHOLD_BETA) \
            if config.USE_OPTIMAL_THRESHOLD else 0.5

        cv_thresholds.append(fold_threshold)
        val_pred = (val_probs > fold_threshold).astype(int)

        fold_metrics = calculate_metrics(val_true, val_pred, val_probs)
        fold_metrics['Fold'] = fold + 1
        fold_metrics['Threshold'] = fold_threshold
        cv_results.append(fold_metrics)

        print(f"  Fold {fold + 1} (t={fold_threshold:.3f}): "
              f"AUC={fold_metrics['ROC-AUC']:.4f}, F1={fold_metrics['F1']:.4f}, "
              f"P={fold_metrics['Precision']:.4f}, R={fold_metrics['Recall']:.4f}")

    # Aggregate
    cv_df = pd.DataFrame(cv_results)

    mean_row = {'Fold': 'Mean', 'Threshold': np.mean(cv_thresholds)}
    std_row = {'Fold': 'Std', 'Threshold': np.std(cv_thresholds)}

    for col in cv_df.columns:
        if col not in ['Fold', 'Threshold']:
            mean_row[col] = cv_df[col].mean()
            std_row[col] = cv_df[col].std()

    cv_df = pd.concat([cv_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print("\n" + "-" * 40)
    print("CV SUMMARY:")
    print("-" * 40)
    for metric in ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall']:
        m = cv_df[cv_df['Fold'] == 'Mean'][metric].values[0]
        s = cv_df[cv_df['Fold'] == 'Std'][metric].values[0]
        print(f"  {metric}: {m:.4f} +/- {s:.4f}")

    return cv_df, cv_thresholds

# ============================================================================
# FINAL TRAINING
# ============================================================================

def train_final_model(X_raw, y_np, full_edge_index, train_val_idx, test_idx, config, device):
    """Train final model - uses FULL graph structure"""
    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING (LABEL-ONLY SPLIT)")
    print("=" * 60)
    print("[TRAIN] Full graph structure is used (standard GNN practice)")

    N = len(y_np)
    trainer = Trainer(config, device)

    ft_idx, fv_idx = train_test_split(
        train_val_idx, test_size=0.1, stratify=y_np[train_val_idx], random_state=config.SEED
    )

    # Leak-free normalization
    X_normalized, scaler = LeakFreeNormalizer.normalize_for_split(X_raw, train_val_idx, test_idx)
    X_feat = torch.tensor(X_normalized, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    # Full graph on device
    train_edge_index = full_edge_index.to(device)
    print(f"[TRAIN] Training with FULL graph: {train_edge_index.shape[1] // 2} edges")

    # Verify no label leakage
    splitter = DataSplitter(N, y_np, full_edge_index, config.SEED)
    splitter.validate_no_label_leakage(train_val_idx, test_idx)

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[ft_idx] = True
    val_mask[fv_idx] = True

    # Model
    model = trainer.create_model(X_feat.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30)
    cls_criterion, supcon_criterion = trainer.create_criterion()

    best_auc, best_threshold, best_state, no_improve = 0, 0.5, None, 0
    history = {'loss': [], 'val_auc': [], 'threshold': []}

    for epoch in range(config.EPOCHS):
        loss = trainer.train_epoch(
            model, optimizer, X_feat, y, train_edge_index,
            train_mask, cls_criterion, supcon_criterion, N
        )

        val_auc, val_probs, val_true = trainer.validate(model, X_feat, y, train_edge_index, val_mask)

        current_threshold = find_optimal_threshold(val_true, val_probs, config.THRESHOLD_BETA) \
            if config.USE_OPTIMAL_THRESHOLD else 0.5 

        scheduler.step(val_auc)
        history['loss'].append(loss)
        history['val_auc'].append(val_auc)
        history['threshold'].append(current_threshold)

        if val_auc > best_auc:
            best_auc = val_auc
            best_threshold = current_threshold
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.PATIENCE:
                print(f"[TRAIN] Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"[TRAIN] Epoch {epoch + 1}: Loss={loss:.4f}, AUC={val_auc:.4f}, t={current_threshold:.3f}")

    print(f"\n[TRAIN] Best: AUC={best_auc:.4f}, threshold={best_threshold:.4f}")

    model.load_state_dict(best_state)
    return model, scaler, best_threshold, history, train_edge_index, X_feat, y

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_test_set(model, X_feat, y, y_np, train_edge_index, test_idx, optimal_threshold, device):
    """Evaluate on test set - uses FULL graph"""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (LABEL-ONLY SPLIT)")
    print("=" * 60)
    print("[EVAL] Test nodes use full graph structure for prediction")

    model.eval()

    with torch.no_grad():
        logits, _, z_proj = model(X_feat, train_edge_index)
        probs = torch.sigmoid(logits)

    y_true = y_np[test_idx]
    y_scores = probs[test_idx].cpu().numpy()
    y_pred = (y_scores > optimal_threshold).astype(int)

    test_metrics = calculate_metrics(y_true, y_pred, y_scores)
    test_metrics['Threshold'] = optimal_threshold

    # Comparison
    y_pred_default = (y_scores > 0.5).astype(int)
    test_metrics_default = calculate_metrics(y_true, y_pred_default, y_scores)

    print(f"\n[EVAL] Threshold: {optimal_threshold:.4f}")
    print(f"\n{'Metric':<12} {'Optimal':>10} {'Default':>10} {'Difference':>10}")
    print("-" * 45)
    for m in ['Precision', 'Recall', 'F1', 'MCC']:
        opt, dflt = test_metrics[m], test_metrics_default[m]
        print(f"{m:<12} {opt:>10.4f} {dflt:>10.4f} {opt - dflt:>+10.4f}")

    return test_metrics, y_true, y_pred, y_scores, z_proj

# ============================================================================
# VISUALIZATION
# ============================================================================
def create_visualizations(y_true, y_pred, y_scores, optimal_threshold,
                          history, z_proj, y_np, test_idx, train_val_idx,
                          cv_df, test_metrics, config):
    """Generate plots"""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    output_dir = config.OUTPUT_DIR

    # 1. ROC, PR, Threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    axes[0].plot(fpr, tpr, 'darkorange', lw=2, label=f"AUC={test_metrics['ROC-AUC']:.4f}")
    axes[0].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    idx = np.argmin(np.abs(thresholds_roc - optimal_threshold))
    axes[0].scatter([fpr[idx]], [tpr[idx]], c='red', s=100, zorder=5, label=f't={optimal_threshold:.3f}')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    axes[1].plot(rec, prec, 'green', lw=2, label=f"AUPR={test_metrics['PR-AUC']:.4f}")
    axes[1].axhline(y_true.mean(), color='navy', linestyle='--', label='Baseline')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('PR Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    ts = np.linspace(0.1, 0.9, 50)
    axes[2].plot(ts, [f1_score(y_true, (y_scores > t).astype(int), zero_division=0) for t in ts],
                 'purple', lw=2, label='F1')
    axes[2].plot(ts, [precision_score(y_true, (y_scores > t).astype(int), zero_division=0) for t in ts],
                 'b--', lw=2, label='Precision')
    axes[2].plot(ts, [recall_score(y_true, (y_scores > t).astype(int), zero_division=0) for t in ts],
                 'g:', lw=2, label='Recall')
    axes[2].axvline(optimal_threshold, color='red', lw=2, label=f'Optimal={optimal_threshold:.3f}')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Metrics vs Threshold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/roc_pr_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved {output_dir}/plots/roc_pr_threshold.png")

    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Non-BM', 'Biomarker']).plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix (t={optimal_threshold:.3f})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/confusion_matrix.png', dpi=300)
    plt.close()
    print(f"[VIZ] Saved {output_dir}/plots/confusion_matrix.png")

    # 3. Training History
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['loss'], 'b', lw=2)
    axes[0].set_title('Loss')
    axes[0].grid(alpha=0.3)
    axes[1].plot(history['val_auc'], 'g', lw=2)
    axes[1].set_title('Val AUC')
    axes[1].grid(alpha=0.3)
    axes[2].plot(history['threshold'], 'r', lw=2)
    axes[2].set_title('Threshold')
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/training_history.png', dpi=300)
    plt.close()
    print(f"[VIZ] Saved {output_dir}/plots/training_history.png")

    # 4. t-SNE
    for idx_set, name in [(test_idx, 'test'), (train_val_idx, 'train')]:
        emb = z_proj[idx_set].cpu().numpy()
        labels = y_np[idx_set]
        if len(emb) >= 30:
            z_2d = TSNE(n_components=2, random_state=config.SEED,
                        perplexity=min(30, len(emb) - 1)).fit_transform(emb)
            fig, ax = plt.subplots(figsize=(10, 8))
            for i, (c, m, l) in enumerate([('blue', 'o', 'Non-BM'), ('red', '^', 'Biomarker')]):
                mask = labels == i
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=c, marker=m, s=50,
                           alpha=0.6, label=l, edgecolors='k', linewidth=0.5)
            ax.set_title(f'{name.capitalize()} Embeddings')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/plots/tsne_{name}.png', dpi=300)
            plt.close()
            print(f"[VIZ] Saved {output_dir}/plots/tsne_{name}.png")

    # 5. CV vs Test
    cv_mean = cv_df[cv_df['Fold'] == 'Mean'].iloc[0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'MCC']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics))
    w = 0.35

    cv_values = [cv_mean[m] for m in metrics]
    test_values = [test_metrics[m] for m in metrics]

    bars1 = ax.bar(x - w / 2, cv_values, w, label='CV Mean', color='skyblue', edgecolor='k')
    bars2 = ax.bar(x + w / 2, test_values, w, label='Test', color='salmon', edgecolor='k')

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'CV vs Test (t={optimal_threshold:.3f})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/metrics_comparison.png', dpi=300)
    plt.close()
    print(f"[VIZ] Saved {output_dir}/plots/metrics_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(model, scaler, optimal_threshold, cv_thresholds, config,
                 test_metrics, cv_df, test_idx, y_true, y_pred, y_scores, nodes,
                 stats_tracker):
    """Save all results"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    output_dir = config.OUTPUT_DIR

    torch.save(model.state_dict(), f'{output_dir}/models/model_weights.pt')
    print(f"[SAVE] {output_dir}/models/model_weights.pt")

    with open(f'{output_dir}/models/config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"[SAVE] {output_dir}/models/config.json")

    threshold_info = {
        'optimal_threshold': optimal_threshold,
        'cv_thresholds': cv_thresholds,
        'cv_threshold_mean': float(np.mean(cv_thresholds)),
        'cv_threshold_std': float(np.std(cv_thresholds))
    }
    with open(f'{output_dir}/models/threshold_info.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"[SAVE] {output_dir}/models/threshold_info.json")

    cv_df.to_csv(f'{output_dir}/metrics/cv_results.csv', index=False)
    print(f"[SAVE] {output_dir}/metrics/cv_results.csv")

    pd.DataFrame([test_metrics]).to_csv(f'{output_dir}/metrics/test_metrics.csv', index=False)
    print(f"[SAVE] {output_dir}/metrics/test_metrics.csv")

    predictions_df = pd.DataFrame({
        'node_idx': test_idx,
        'gene_symbol': nodes.iloc[test_idx]['SYMBOL'].values,
        'true_label': y_true,
        'predicted_label': y_pred,
        'predicted_score': y_scores
    }).sort_values('predicted_score', ascending=False)
    predictions_df.to_csv(f'{output_dir}/metrics/test_predictions.csv', index=False)
    print(f"[SAVE] {output_dir}/metrics/test_predictions.csv")

    top = predictions_df[predictions_df['predicted_label'] == 1].head(20)
    top.to_csv(f'{output_dir}/metrics/top_biomarker_candidates.csv', index=False)
    print(f"[SAVE] {output_dir}/metrics/top_biomarker_candidates.csv")

    # Save graph statistics
    stats_tracker.save_to_csv(f'{output_dir}/metrics/graph_statistics.csv')

    return predictions_df, top

# =========================================
# MAIN
# =========================================
def main():
    print("=" * 60)
    print("BIOMARKER PREDICTION WITH GNN")
    print("=" * 60)

    Config.validate()
    device = setup_environment(Config.SEED)
    create_output_dirs(Config.OUTPUT_DIR)

    # Initialize graph statistics tracker
    stats_tracker = GraphStatisticsTracker()

    print("\n" + "-" * 40)
    print("LOADING DATA")
    print("-" * 40)

    loader = DataLoader(Config)
    geokg, go_terms = loader.load_features()
    nodes = loader.load_nodes()
    edges = loader.load_edges()

    # Record initial state
    stats_tracker.add_stage(
        stage_name="Initial Load",
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_positives=nodes["is_biomarker"].sum(),
        additional_info=f"geokg_shape={geokg.shape}, go_terms_shape={go_terms.shape}"
    )

    # Filter invalid features
    geokg, go_terms, nodes, valid_mask, n_removed_features = loader.filter_invalid_features(
        geokg, go_terms, nodes
    )
    stats_tracker.add_stage(
        stage_name="After Feature Filtering",
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_positives=nodes["is_biomarker"].sum(),
        n_removed_nodes=n_removed_features,
        removal_reason="Invalid features (zero norm)"
    )

    X_raw = np.hstack([geokg, go_terms])
    y_np = nodes["is_biomarker"].values.astype(int)

    # Build edge index (CPU first)
    full_edge_index, sym2idx = loader.build_edge_index(nodes, edges)

    stats_tracker.add_stage(
        stage_name="After Edge Index Build",
        n_nodes=len(y_np),
        n_edges=full_edge_index.shape[1] // 2,
        n_positives=y_np.sum()
    )

    # Remove isolated nodes
    X_raw, y_np, nodes, full_edge_index = remove_isolated_nodes(
        X_raw, y_np, nodes, full_edge_index, stats_tracker
    )

    N = len(y_np)
    print(f"\n[DATA] Final dataset: {N} nodes, {y_np.sum()} positives ({100 * y_np.mean():.1f}%)")

    stats_tracker.add_stage(
        stage_name="Final Processing",
        n_nodes=N,
        n_edges=full_edge_index.shape[1] // 2,
        n_positives=y_np.sum()
    )

    # Move edge index to device
    full_edge_index = full_edge_index.to(device)

    print("\n" + "-" * 40)
    print("DATA SPLITTING")
    print("-" * 40)

    splitter = DataSplitter(N, y_np, full_edge_index, Config.SEED)
    train_val_idx, test_idx = splitter.train_test_split(test_size=0.2)

    # Count edges for each split
    train_val_edges = splitter.count_edges_for_split(train_val_idx)
    test_edges = splitter.count_edges_for_split(test_idx)

    stats_tracker.add_stage(
        stage_name="Train/Val Split",
        n_nodes=len(train_val_idx),
        n_edges=train_val_edges,
        n_positives=y_np[train_val_idx].sum(),
        additional_info="Edges within train/val node set only"
    )

    stats_tracker.add_stage(
        stage_name="Test Split",
        n_nodes=len(test_idx),
        n_edges=test_edges,
        n_positives=y_np[test_idx].sum(),
        additional_info="Edges within test node set only"
    )

    cv_df, cv_thresholds = run_cross_validation(
        X_raw, y_np, full_edge_index, train_val_idx, Config, device
    )

    model, scaler, optimal_threshold, history, train_edge_index, X_feat, y = train_final_model(
        X_raw, y_np, full_edge_index, train_val_idx, test_idx, Config, device
    )

    test_metrics, y_true, y_pred, y_scores, z_proj = evaluate_test_set(
        model, X_feat, y, y_np, train_edge_index, test_idx, optimal_threshold, device
    )

    create_visualizations(
        y_true, y_pred, y_scores, optimal_threshold, history, z_proj,
        y_np, test_idx, train_val_idx, cv_df, test_metrics, Config
    )

    predictions_df, top = save_results(
        model, scaler, optimal_threshold, cv_thresholds, Config,
        test_metrics, cv_df, test_idx, y_true, y_pred, y_scores, nodes,
        stats_tracker
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n[RESULT] Threshold: {optimal_threshold:.4f} "
          f"(CV: {np.mean(cv_thresholds):.4f} +/- {np.std(cv_thresholds):.4f})")

    print(f"\n[RESULT] CV Results:")
    for m in ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall']:
        mean = cv_df[cv_df['Fold'] == 'Mean'][m].values[0]
        std = cv_df[cv_df['Fold'] == 'Std'][m].values[0]
        print(f"   {m}: {mean:.4f} +/- {std:.4f}")

    print(f"\n[RESULT] Test Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")

    print(f"\n[RESULT] Top 5 Candidates:")
    for _, row in top.head(5).iterrows():
        status = "CORRECT" if row['true_label'] == 1 else "FP"
        print(f"   [{status}] {row['gene_symbol']}: {row['predicted_score']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()