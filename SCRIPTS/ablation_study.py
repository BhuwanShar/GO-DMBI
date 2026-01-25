import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, matthews_corrcoef, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import copy
import json
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

# ============================
# SETUP
# ============================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "update_ablation_study_glio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")

# ===========================================
# CONFIGURATION 
# ===========================================
class AblationConfig:
    
    # Data paths - MATCHING MAIN CODE
    GEOKG_PATH = "DATA/Data_process_output/glio_node_features_geokg.npy"
    GO_TERMS_PATH = "DATA/Data_process_output/glio_node_features_go_terms.npy"
    NODES_PATH = "DATA/Data_process_output/glio_nodes_with_uniprot_and_embidx.csv"
    EDGES_PATH = "DATA/Raw/Glioblastoma_microarray/glio_edges.csv"
    
    
    # Model hyperparameters - MATCHING MAIN CODE
    HIDDEN_DIM = 512
    PROJ_DIM = 64
    NUM_GRAPH_LAYERS = 16
    DROPOUT = 0.3
    
    # Training hyperparameters
    LR = 0.0003
    WEIGHT_DECAY = 0.01
    EPOCHS = 500
    PATIENCE = 50
    DROP_EDGE_RATE = 0.3
    
    # Loss function
    FOCAL_GAMMA = 4.3
    POS_WEIGHT = 4
    FP_WEIGHT = 0.3
    
    # Contrastive learning
    SUPCON_TEMP = 0.06
    LAMBDA_SUPCON = 0.34
    
    # Latent SMOTE
    SMOTE_K = 6
    SMOTE_ALPHA = 1.9
    
    # Threshold
    THRESHOLD_BETA = 0.9
    
    # Activations
    MLP_ACTIVATION = 'relu'
    GRAPH_ACTIVATION = 'gelu'
    PROJ_ACTIVATION = 'leaky_relu'
    
    # Ablation study specific
    N_RUNS = 3  # Number of runs for averaging results
    
    @classmethod
    def to_dict(cls):
        # Convert config to dictionary for json
        result = {}
        for k, v in cls.__dict__.items():
            # Skip private/magic attributes
            if k.startswith('_'):
                continue
            # Skip methods, classmethods, staticmethods
            if isinstance(v, (classmethod, staticmethod)):
                continue
            if callable(v):
                continue
            # Only include JSON-serializable types
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                result[k] = v
        return result


# ===================================================
# DATA LOADING 
# ===================================================
def load_data(config):
    # Load and prepare data 
    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    # Load features
    geokg = np.load(config.GEOKG_PATH)
    go_terms = np.load(config.GO_TERMS_PATH)
    nodes = pd.read_csv(config.NODES_PATH)
    edges = pd.read_csv(config.EDGES_PATH)
    
    print(f"Initial: nodes={len(nodes)}, edges={len(edges)}")
    print(f"Features: geokg={geokg.shape}, go_terms={go_terms.shape}")
    
    # Filter invalid features (zero norm)
    valid_mask = np.linalg.norm(geokg, axis=1) > 1e-6
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"Removed {n_invalid} nodes with invalid features")
    
    geokg = geokg[valid_mask]
    go_terms = go_terms[valid_mask]
    nodes = nodes.loc[valid_mask].reset_index(drop=True)
    
    y_np = nodes["is_biomarker"].values.astype(int)
    
    # Build edge index
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
        print(f"Skipped {skipped} edges (nodes not found)")
    
    if len(valid_edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    
    print(f"After feature filtering: {len(y_np)} nodes, {edge_index.shape[1]//2} undirected edges")
    
    return geokg, go_terms, y_np, edge_index, nodes


def remove_isolated_nodes(X, y_np, nodes, edge_index):
    # Remove nodes with degree 0 
    n_nodes = len(y_np)
    
    if edge_index.numel() == 0:
        degrees = np.zeros(n_nodes, dtype=np.int64)
    else:
        edge_index_np = edge_index.cpu().numpy()
        degrees = np.zeros(n_nodes, dtype=np.int64)
        np.add.at(degrees, edge_index_np[0], 1)
    
    isolated_mask = degrees == 0
    n_isolated = isolated_mask.sum()
    
    if n_isolated == 0:
        print("No isolated nodes found")
        return X, y_np, nodes, edge_index
    
    print(f"Removing {n_isolated} isolated nodes (degree=0)")
    
    # Get non-isolated indices
    keep_mask = ~isolated_mask
    keep_indices = np.where(keep_mask)[0]
    
    # Mapping from old to new indices
    old_to_new = np.full(n_nodes, -1, dtype=np.int64)
    old_to_new[keep_indices] = np.arange(len(keep_indices))
    
    # Filter
    X_filtered = X[keep_mask]
    y_filtered = y_np[keep_mask]
    nodes_filtered = nodes.loc[keep_mask].reset_index(drop=True)
    
    # Remap edges
    edge_index_np = edge_index.cpu().numpy()
    new_sources = old_to_new[edge_index_np[0]]
    new_targets = old_to_new[edge_index_np[1]]
    
    valid_edge_mask = (new_sources >= 0) & (new_targets >= 0)
    new_edge_index = torch.tensor(
        np.stack([new_sources[valid_edge_mask], new_targets[valid_edge_mask]]),
        dtype=torch.long
    )
    
    removed_biomarkers = nodes.loc[isolated_mask, 'is_biomarker'].sum()
    print(f"Removed {n_isolated} nodes ({removed_biomarkers} were biomarkers)")
    
    return X_filtered, y_filtered, nodes_filtered, new_edge_index


def prepare_features(geokg, go_terms, feature_type='concat'):
    # Prepare features based on ablation type
    if feature_type == 'geokg_only':
        return geokg.copy()
    elif feature_type == 'go_only':
        return go_terms.copy()
    else:  # concat
        return np.hstack([geokg, go_terms])


# ===================================
# UTILITY FUNCTIONS
# ===================================
def get_activation(name):
    # Return activation function by name
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
    # Find threshold maximizing F-beta
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

# =============================
# LOSS FUNCTIONS
# ==============================
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
    """Supervised contrastive loss"""
    
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

# =====================================================
# MODEL 
# =====================================================
class GCNLayer(nn.Module):
    
    def __init__(self, in_feat, out_feat, dropout=0.3, activation='gelu'):
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


class AblationGNN(nn.Module):
    # GNN for biomarker prediction with ablation flags 
    def __init__(self, n_feat, config, 
                 use_mlp=True, 
                 use_gcn=True, 
                 use_gated_fusion=True,
                 num_gcn_layers=None):
        super().__init__()
        self.config = config
        self.use_mlp = use_mlp
        self.use_gcn = use_gcn
        self.use_gated_fusion = use_gated_fusion
        
        hidden_dim = config.HIDDEN_DIM
        dropout = config.DROPOUT
        num_layers = num_gcn_layers if num_gcn_layers else config.NUM_GRAPH_LAYERS
        
        # MLP branch
        if use_mlp:
            self.feat_mlp = nn.Sequential(
                nn.Linear(n_feat, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                get_activation(config.MLP_ACTIVATION),
                nn.Dropout(dropout)
            )
        
        # GCN branch
        if use_gcn:
            self.graph_layers = nn.ModuleList()
            self.graph_layers.append(
                GCNLayer(n_feat, hidden_dim, dropout, config.GRAPH_ACTIVATION)
            )
            for _ in range(num_layers - 1):
                self.graph_layers.append(
                    GCNLayer(hidden_dim, hidden_dim, dropout, config.GRAPH_ACTIVATION)
                )
        
        # Gated fusion
        if use_mlp and use_gcn and use_gated_fusion:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, 1),
                nn.Sigmoid()
            )
        
        # Determine output dimension
        if use_mlp and use_gcn:
            fused_dim = hidden_dim
        elif use_mlp:
            fused_dim = hidden_dim
        elif use_gcn:
            fused_dim = hidden_dim
        else:
            raise ValueError("At least one of MLP or GCN must be enabled")
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            get_activation(config.PROJ_ACTIVATION),
            nn.Linear(fused_dim, config.PROJ_DIM)
        )
        
        # Classifier
        self.classifier = nn.Linear(fused_dim, 1)
    
    def forward(self, x, edge_index, labels=None, use_smote=False):
        h_mlp = None
        h_graph = None
        
        # MLP branch
        if self.use_mlp:
            h_mlp = self.feat_mlp(x)
        
        # GCN branch
        if self.use_gcn:
            h_graph = x
            for layer in self.graph_layers:
                h_graph = layer(h_graph, edge_index)
        
        # Fusion
        if self.use_mlp and self.use_gcn:
            if self.use_gated_fusion:
                combined = torch.cat([h_mlp, h_graph], dim=1)
                gate = self.gate(combined)
                h_fused = gate * h_graph + (1 - gate) * h_mlp
            else:
                # Simple average fusion
                h_fused = (h_mlp + h_graph) / 2
        elif self.use_mlp:
            h_fused = h_mlp
        else:
            h_fused = h_graph
        
        # Latent SMOTE
        if use_smote and labels is not None and self.training:
            h_fused, labels = latent_smote(
                h_fused, labels, 
                k=self.config.SMOTE_K, 
                alpha=self.config.SMOTE_ALPHA
            )
        
        z_proj = self.projection(h_fused)
        logits = self.classifier(h_fused).view(-1)
        
        return logits, labels, z_proj

# ======================================
# METRICS CALCULATION
# ======================================
def calculate_metrics(y_true, y_pred, y_scores):
    # Calculate evaluation metrics
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
        'PR-AUC': average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

# ========================================================
# TRAINING FUNCTION
# ========================================================
def run_experiment(X_raw, y_np, full_edge_index, config, 
                   use_mlp=True, 
                   use_gcn=True, 
                   use_smote=True, 
                   use_supcon=True,
                   use_focal=True,
                   use_edge_drop=True,
                   use_gated_fusion=True,
                   num_gcn_layers=None,
                   run_id=0):
    
    N = len(y_np)
    
    # Split node indices 
    all_idx = np.arange(N)
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, stratify=y_np, random_state=SEED + run_id
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1, stratify=y_np[train_val_idx], random_state=SEED + run_id
    )
    
    # Verify no label leakage
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val overlap!"
    assert len(set(train_val_idx) & set(test_idx)) == 0, "Train+Val/Test overlap!"
    
    # Normalize features (fit on train, transform all)
    scaler = StandardScaler()
    X_norm = np.zeros_like(X_raw)
    X_norm[train_val_idx] = scaler.fit_transform(X_raw[train_val_idx])
    X_norm[test_idx] = scaler.transform(X_raw[test_idx])
    
    X = torch.tensor(X_norm, dtype=torch.float, device=DEVICE)
    y = torch.tensor(y_np, dtype=torch.float, device=DEVICE)
    edge_index = full_edge_index.to(DEVICE)
    
    # Create masks
    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Model
    model = AblationGNN(
        X.shape[1], config, 
        use_mlp=use_mlp, 
        use_gcn=use_gcn,
        use_gated_fusion=use_gated_fusion,
        num_gcn_layers=num_gcn_layers
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Loss functions
    if use_focal:
        criterion_cls = FocalLoss(gamma=config.FOCAL_GAMMA, fp_weight=config.FP_WEIGHT)
    else:
        criterion_cls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.POS_WEIGHT], device=DEVICE)
        )
    
    criterion_supcon = SupervisedContrastiveLoss(temperature=config.SUPCON_TEMP)
    
    # Training
    best_auc, best_state, patience_cnt = 0, None, 0
    drop_rate = config.DROP_EDGE_RATE if use_edge_drop else 0
    
    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Drop edges for regularization
        batch_edge = drop_edge(edge_index, drop_rate, True)
        
        # Forward pass
        logits, targets, z_proj = model(X, batch_edge, y, use_smote=use_smote)
        
        # Compute loss on training nodes only
        num_syn = logits.shape[0] - N
        train_indices = torch.nonzero(train_mask).squeeze(-1)
        
        if num_syn > 0:
            synthetic_indices = torch.arange(N, N + num_syn, device=DEVICE)
            active_idx = torch.cat([train_indices, synthetic_indices])
        else:
            active_idx = train_indices
        
        loss = criterion_cls(logits[active_idx], targets[active_idx])
        
        if use_supcon:
            loss += config.LAMBDA_SUPCON * criterion_supcon(
                z_proj[active_idx], targets[active_idx]
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, _, _ = model(X, edge_index, None, use_smote=False)
            v_probs = torch.sigmoid(v_logits[val_mask]).cpu().numpy()
            v_true = y[val_mask].cpu().numpy()
            
            try:
                v_auc = roc_auc_score(v_true, v_probs)
            except ValueError:
                v_auc = 0.5
        
        if v_auc > best_auc:
            best_auc = v_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config.PATIENCE:
                break
    
    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        logits, _, _ = model(X, edge_index, None, use_smote=False)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    y_true = y_np[test_idx]
    y_scores = probs[test_idx]
    
    # Find optimal threshold
    threshold = find_optimal_threshold(y_true, y_scores, config.THRESHOLD_BETA)
    y_pred = (y_scores > threshold).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    metrics['Threshold'] = threshold
    
    return metrics

def run_experiment_with_averaging(X_raw, y_np, full_edge_index, config, 
                                  n_runs=3, **ablation_flags):
    """Run experiment multiple times and average results"""
    all_metrics = []
    
    for run_id in range(n_runs):
        metrics = run_experiment(
            X_raw, y_np, full_edge_index, config,
            run_id=run_id,
            **ablation_flags
        )
        all_metrics.append(metrics)
    
    # Average results
    avg_metrics = {}
    std_metrics = {}
    
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    return avg_metrics, std_metrics, all_metrics

# ===================================
# ABLATION STUDY
# ===================================
def run_ablation_study(config):
    # Run complete ablation study
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY - BIOMARKER GNN")
    print("=" * 70)
    print("Using FULL graph structure for all experiments (standard GNN practice)")
    print(f"Each experiment averaged over {config.N_RUNS} runs")
    
    # Load data
    geokg, go_terms, y_np, full_edge_index, nodes = load_data(config)
    
    # Remove isolated nodes
    X_concat_temp = np.hstack([geokg, go_terms])
    X_temp, y_np, nodes, full_edge_index = remove_isolated_nodes(
        X_concat_temp, y_np, nodes, full_edge_index
    )
    
    # Reload features after filtering
    geokg = X_temp[:, :geokg.shape[1]]
    go_terms = X_temp[:, geokg.shape[1]:]
    
    N = len(y_np)
    print(f"\nFinal dataset: {N} nodes, {y_np.sum()} positives ({100*y_np.mean():.1f}%)")
    print(f"Full graph: {full_edge_index.shape[1]//2} undirected edges")
    
    results = {}
    std_results = {}
    
    # ========== FEATURE ABLATION ==========
    print("\n" + "-" * 50)
    print(" FEATURE ABLATION")
    print("-" * 50)
    
    feature_configs = {
        'GeoKG Only': 'geokg_only',
        'GO Terms Only': 'go_only',
        'Concatenated (Full)': 'concat'
    }
    
    for name, feat_type in feature_configs.items():
        print(f"\n  Running: {name}...")
        X = prepare_features(geokg, go_terms, feat_type)
        
        avg_metrics, std_metrics, _ = run_experiment_with_averaging(
            X, y_np, full_edge_index, config, n_runs=config.N_RUNS,
            use_mlp=True, use_gcn=True, use_smote=True, 
            use_supcon=True, use_focal=True, use_edge_drop=True,
            use_gated_fusion=True
        )
        
        results[name] = avg_metrics
        std_results[name] = std_metrics
        print(f"    ROC-AUC: {avg_metrics['ROC-AUC']:.4f} ± {std_metrics['ROC-AUC']:.4f}")
        print(f"    F1: {avg_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
    
    # ========== MODEL COMPONENT ABLATION ==========
    print("\n" + "-" * 50)
    print(" MODEL COMPONENT ABLATION")
    print("-" * 50)
    
    # Use concatenated features for model ablation
    X_concat = prepare_features(geokg, go_terms, 'concat')
    
    model_configs = {
        'Full Model': {
            'use_mlp': True, 'use_gcn': True, 'use_smote': True, 
            'use_supcon': True, 'use_focal': True, 'use_edge_drop': True,
            'use_gated_fusion': True
        },
        'Without MLP': {
            'use_mlp': False, 'use_gcn': True, 'use_smote': True, 
            'use_supcon': True, 'use_focal': True, 'use_edge_drop': True,
            'use_gated_fusion': True
        },
        'Without GCN': {
            'use_mlp': True, 'use_gcn': False, 'use_smote': True, 
            'use_supcon': True, 'use_focal': True, 'use_edge_drop': True,
            'use_gated_fusion': True
        },
        'Without SMOTE': {
            'use_mlp': True, 'use_gcn': True, 'use_smote': False, 
            'use_supcon': True, 'use_focal': True, 'use_edge_drop': True,
            'use_gated_fusion': True
        },
        'Without SupCon': {
            'use_mlp': True, 'use_gcn': True, 'use_smote': True, 
            'use_supcon': False, 'use_focal': True, 'use_edge_drop': True,
            'use_gated_fusion': True
        },
        'Without Focal Loss': {
            'use_mlp': True, 'use_gcn': True, 'use_smote': True, 
            'use_supcon': True, 'use_focal': False, 'use_edge_drop': True,
            'use_gated_fusion': True
        },
        'Without Edge Dropout': {
            'use_mlp': True, 'use_gcn': True, 'use_smote': True, 
            'use_supcon': True, 'use_focal': True, 'use_edge_drop': False,
            'use_gated_fusion': True
        },
        'Without Gated Fusion': {
            'use_mlp': True, 'use_gcn': True, 'use_smote': True, 
            'use_supcon': True, 'use_focal': True, 'use_edge_drop': True,
            'use_gated_fusion': False
        },
    }
    
    for name, cfg in model_configs.items():
        print(f"\n  Running: {name}...")
        
        avg_metrics, std_metrics, _ = run_experiment_with_averaging(
            X_concat, y_np, full_edge_index, config, n_runs=config.N_RUNS,
            **cfg
        )
        
        results[name] = avg_metrics
        std_results[name] = std_metrics
        print(f"    ROC-AUC: {avg_metrics['ROC-AUC']:.4f} ± {std_metrics['ROC-AUC']:.4f}")
        print(f"    F1: {avg_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
    
    # ========== GCN DEPTH ABLATION ==========
    print("\n" + "-" * 50)
    print(" GCN DEPTH ABLATION")
    print("-" * 50)
    
    layer_configs = [2, 4, 8, 12, 16, 20]
    
    for n_layers in layer_configs:
        name = f'{n_layers} GCN Layers'
        print(f"\n  Running: {name}...")
        
        avg_metrics, std_metrics, _ = run_experiment_with_averaging(
            X_concat, y_np, full_edge_index, config, n_runs=config.N_RUNS,
            use_mlp=True, use_gcn=True, use_smote=True, 
            use_supcon=True, use_focal=True, use_edge_drop=True,
            use_gated_fusion=True, num_gcn_layers=n_layers
        )
        
        results[name] = avg_metrics
        std_results[name] = std_metrics
        print(f"    ROC-AUC: {avg_metrics['ROC-AUC']:.4f} ± {std_metrics['ROC-AUC']:.4f}")
        print(f"    F1: {avg_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
    
    return results, std_results

# ============================================================================
# VISUALIZATION
# ============================================================================
def create_visualizations(results, std_results, config):
    # Create ablation study visualizations 
    
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    # Convert to DataFrames
    df = pd.DataFrame(results).T
    df_std = pd.DataFrame(std_results).T
    
    metrics = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    # ========== HEATMAP ==========
    # Separate sections
    feature_rows = ['GeoKG Only', 'GO Terms Only', 'Concatenated (Full)']
    component_rows = ['Full Model', 'Without MLP', 'Without GCN', 'Without SMOTE', 
                      'Without SupCon', 'Without Focal Loss', 'Without Edge Dropout',
                      'Without Gated Fusion']
    depth_rows = [f'{n} GCN Layers' for n in [2, 4, 8, 12, 16, 20]]
    
    # Combined heatmap
    all_rows = feature_rows + component_rows + depth_rows
    existing_rows = [r for r in all_rows if r in df.index]
    
    fig, ax = plt.subplots(figsize=(14, 16))
    
    plot_df = df.loc[existing_rows, metrics]
    
    sns.heatmap(plot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Score'})
    
    # Add separator lines
    if len(feature_rows) <= len(existing_rows):
        ax.axhline(y=len(feature_rows), color='black', linewidth=3)
    if len(feature_rows) + len(component_rows) <= len(existing_rows):
        ax.axhline(y=len(feature_rows) + len(component_rows), color='black', linewidth=3)
    
    ax.set_title('Ablation Study Results - Biomarker GNN', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('')
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/ablation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/plots/ablation_heatmap.png")
    
    return df

# ===========================================
# SAVE RESULTS
# ===========================================
def save_results(results, std_results, config):
    # Save ablation study results
    
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)
    
    # Convert to DataFrames
    df = pd.DataFrame(results).T
    df_std = pd.DataFrame(std_results).T
    
    # Save main results
    df.to_csv(f'{OUTPUT_DIR}/metrics/ablation_results.csv')
    print(f"  Saved: {OUTPUT_DIR}/metrics/ablation_results.csv")
    
    # Save standard deviations
    df_std.to_csv(f'{OUTPUT_DIR}/metrics/ablation_results_std.csv')
    print(f"  Saved: {OUTPUT_DIR}/metrics/ablation_results_std.csv")
    
    # Save combined (mean ± std)
    combined = df.copy()
    for col in df.columns:
        combined[col] = df[col].apply(lambda x: f"{x:.4f}") + " ± " + df_std[col].apply(lambda x: f"{x:.4f}")
    combined.to_csv(f'{OUTPUT_DIR}/metrics/ablation_results_combined.csv')
    print(f"  Saved: {OUTPUT_DIR}/metrics/ablation_results_combined.csv")
    
    # Save config - FIXED VERSION
    config_dict = {
        'GEOKG_PATH': config.GEOKG_PATH,
        'GO_TERMS_PATH': config.GO_TERMS_PATH,
        'NODES_PATH': config.NODES_PATH,
        'EDGES_PATH': config.EDGES_PATH,
        'HIDDEN_DIM': config.HIDDEN_DIM,
        'PROJ_DIM': config.PROJ_DIM,
        'NUM_GRAPH_LAYERS': config.NUM_GRAPH_LAYERS,
        'DROPOUT': config.DROPOUT,
        'LR': config.LR,
        'WEIGHT_DECAY': config.WEIGHT_DECAY,
        'EPOCHS': config.EPOCHS,
        'PATIENCE': config.PATIENCE,
        'DROP_EDGE_RATE': config.DROP_EDGE_RATE,
        'FOCAL_GAMMA': config.FOCAL_GAMMA,
        'POS_WEIGHT': config.POS_WEIGHT,
        'FP_WEIGHT': config.FP_WEIGHT,
        'SUPCON_TEMP': config.SUPCON_TEMP,
        'LAMBDA_SUPCON': config.LAMBDA_SUPCON,
        'SMOTE_K': config.SMOTE_K,
        'SMOTE_ALPHA': config.SMOTE_ALPHA,
        'THRESHOLD_BETA': config.THRESHOLD_BETA,
        'MLP_ACTIVATION': config.MLP_ACTIVATION,
        'GRAPH_ACTIVATION': config.GRAPH_ACTIVATION,
        'PROJ_ACTIVATION': config.PROJ_ACTIVATION,
        'N_RUNS': config.N_RUNS,
    }
    
    with open(f'{OUTPUT_DIR}/metrics/config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR}/metrics/config.json")
    
    return df

# =======================================
# PRINT SUMMARY
# =======================================
def print_summary(results, std_results):
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    
    feature_rows = ['GeoKG Only', 'GO Terms Only', 'Concatenated (Full)']
    component_rows = ['Full Model', 'Without MLP', 'Without GCN', 'Without SMOTE', 
                      'Without SupCon', 'Without Focal Loss', 'Without Edge Dropout',
                      'Without Gated Fusion']
    depth_rows = [f'{n} GCN Layers' for n in [2, 4, 8, 12, 16, 20]]
    
    print("\n FEATURE ABLATION:")
    print("-" * 70)
    for name in feature_rows:
        if name in results:
            r = results[name]
            s = std_results[name]
            print(f"  {name:25s} | AUC: {r['ROC-AUC']:.4f}±{s['ROC-AUC']:.4f} | "
                  f"F1: {r['F1']:.4f}±{s['F1']:.4f} | MCC: {r['MCC']:.4f}±{s['MCC']:.4f}")
    
    print("\n MODEL COMPONENT ABLATION:")
    print("-" * 70)
    baseline = results.get('Full Model', {})
    
    for name in component_rows:
        if name in results:
            r = results[name]
            s = std_results[name]
            
            if name == 'Full Model':
                print(f"  {name:25s} | AUC: {r['ROC-AUC']:.4f}±{s['ROC-AUC']:.4f} | "
                      f"F1: {r['F1']:.4f}±{s['F1']:.4f} | (Baseline)")
            else:
                delta_auc = r['ROC-AUC'] - baseline.get('ROC-AUC', 0)
                delta_f1 = r['F1'] - baseline.get('F1', 0)
                sign_auc = "+" if delta_auc >= 0 else ""
                sign_f1 = "+" if delta_f1 >= 0 else ""
                
                print(f"  {name:25s} | AUC: {r['ROC-AUC']:.4f} ({sign_auc}{delta_auc:.4f}) | "
                      f"F1: {r['F1']:.4f} ({sign_f1}{delta_f1:.4f})")
    
    print("\n GCN DEPTH ABLATION:")
    print("-" * 70)
    for name in depth_rows:
        if name in results:
            r = results[name]
            s = std_results[name]
            delta_auc = r['ROC-AUC'] - baseline.get('ROC-AUC', 0) if baseline else 0
            delta_f1 = r['F1'] - baseline.get('F1', 0) if baseline else 0
            
            print(f"  {name:25s} | AUC: {r['ROC-AUC']:.4f}±{s['ROC-AUC']:.4f} | "
                  f"F1: {r['F1']:.4f}±{s['F1']:.4f}")
    
    # Component importance ranking
    print("\n COMPONENT IMPORTANCE (by F1 drop when removed):")
    print("-" * 70)
    
    if baseline:
        importance = []
        for name in component_rows[1:]:  # Skip 'Full Model'
            if name in results:
                delta_f1 = baseline['F1'] - results[name]['F1']
                component = name.replace('Without ', '')
                importance.append((component, delta_f1))
        
        importance.sort(key=lambda x: x[1], reverse=True)
        for i, (component, delta) in enumerate(importance, 1):
            status = "improving" if delta > 0 else "decreasing"
            print(f"  {i}. {component:20s} | F1 drop: {delta:+.4f} {status}")

# =============================================
# MAIN
# =============================================
def main():
    
    print("=" * 70)
    print("BIOMARKER GNN - COMPREHENSIVE ABLATION STUDY")
    print("=" * 70)
    print(f"\nDataset: Glioblastoma")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}/")
    
    config = AblationConfig()
    
    # Run ablation study
    results, std_results = run_ablation_study(config)
    
    # Create visualizations (only heatmap)
    create_visualizations(results, std_results, config)
    
    # Save results
    save_results(results, std_results, config)
    
    # Print summary
    print_summary(results, std_results)
    
    print("\n" + "=" * 70)
    print(" ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - metrics/ablation_results.csv")
    print(f"  - metrics/ablation_results_combined.csv")
    print(f"  - plots/ablation_heatmap.png")


if __name__ == "__main__":
    main()