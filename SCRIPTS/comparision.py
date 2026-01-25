import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, accuracy_score, precision_score, f1_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import copy

warnings.filterwarnings("ignore")

# =====================================
# SETUP
# =====================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "model_comparison_outputs"
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/metrics", exist_ok=True)

print(f"Device: {DEVICE}")

# Configuration dictionary
CONFIG = {
    'hidden_dim': 512,
    'dropout': 0.3,
    'lr': 0.003,
    'weight_decay': 0.01,
    'epochs': 300,
    'patience': 30,
    'n_folds': 5,
    'smote_k': 6,
    'smote_alpha': 1.9,
    'use_smote': True,
}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    
    geokg = np.load("DATA/Data_process_output/glio_node_features_geokg.npy")
    go_terms = np.load("DATA/Data_process_output/glio_node_features_go_terms.npy")
    nodes = pd.read_csv("DATA/Data_process_output/glio_nodes_with_uniprot_and_embidx.csv")
    edges = pd.read_csv("DATA/Raw/Glioblastoma_microarray/glio_edges.csv")
    
    # Filter valid nodes
    valid_mask = np.linalg.norm(geokg, axis=1) > 1e-6
    geokg, go_terms = geokg[valid_mask], go_terms[valid_mask]
    nodes = nodes.loc[valid_mask].reset_index(drop=True)
    
    X_raw = np.hstack([geokg, go_terms])
    y_np = nodes["is_biomarker"].values.astype(int)
    
    # Build edge index
    sym2idx = {s.upper(): i for i, s in enumerate(nodes["SYMBOL"])}
    valid_edges = []
    for _, r in edges.iterrows():
        s, t = r["source"].upper(), r["target"].upper()
        if s in sym2idx and t in sym2idx:
            u, v = sym2idx[s], sym2idx[t]
            if u != v:
                valid_edges.extend([[u, v], [v, u]])
    
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous().to(DEVICE)
    
    pos_count = y_np.sum()
    neg_count = len(y_np) - pos_count
    print(f"Nodes: {len(y_np)}, Positives: {pos_count} ({100*pos_count/len(y_np):.1f}%), "
          f"Negatives: {neg_count} ({100*neg_count/len(y_np):.1f}%)")
    print(f"Edges: {edge_index.shape[1]//2} undirected")
    print(f"Class imbalance ratio: 1:{neg_count/pos_count:.1f}")
    
    return X_raw, y_np, edge_index


def normalize_features(X_raw, train_idx, test_idx):
    '''
    Leak-free feature normalization.
    Fits scaler on training data only, then transforms all data.
    '''
    scaler = StandardScaler()
    X = np.zeros_like(X_raw)
    X[train_idx] = scaler.fit_transform(X_raw[train_idx])
    X[test_idx] = scaler.transform(X_raw[test_idx])
    return X


# ===============================
# SMOTE IMPLEMENTATION
# ===============================
class SMOTESampler:
    
    def __init__(self, k_neighbors=5, alpha=1.0, random_state=None):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.random_state = random_state
        
    def fit_resample_numpy(self, X, y):
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Find minority and majority classes
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_count = counts.max()
        minority_count = counts.min()
        
        # Get minority samples
        minority_idx = np.where(y == minority_class)[0]
        X_minority = X[minority_idx]
        
        # Number of samples to generate
        n_synthetic = majority_count - minority_count
        
        if n_synthetic <= 0 or len(X_minority) < self.k_neighbors + 1:
            print(f"    [SMOTE] No oversampling needed or not enough minority samples")
            return X, y
        
        # Fit nearest neighbors on minority class
        k = min(self.k_neighbors, len(X_minority) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_minority)
        
        # Generate synthetic samples
        synthetic_samples = []
        for _ in range(n_synthetic):
            idx = np.random.randint(len(X_minority))
            sample = X_minority[idx]
            
            _, neighbors = nn.kneighbors([sample])
            neighbor_idx = neighbors[0, np.random.randint(1, k + 1)]
            neighbor = X_minority[neighbor_idx]
            
            lam = np.random.random() * self.alpha
            synthetic = sample + lam * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(n_synthetic, minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])
        
        print(f"    [SMOTE] Generated {n_synthetic} synthetic samples. "
              f"New balance: {np.sum(y_resampled == 0)}:{np.sum(y_resampled == 1)}")
        
        return X_resampled, y_resampled
    
    def fit_resample_latent(self, z, y):
        
        pos_indices = (y == 1).nonzero(as_tuple=True)[0]
        neg_indices = (y == 0).nonzero(as_tuple=True)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        if n_pos >= n_neg or n_pos < self.k_neighbors + 1:
            return z, y
        
        n_synthetic = n_neg - n_pos
        pos_emb = z[pos_indices]
        
        # Find k nearest neighbors
        dist = torch.cdist(pos_emb, pos_emb)
        dist.fill_diagonal_(float('inf'))
        k = min(self.k_neighbors, n_pos - 1)
        _, nn_indices = dist.topk(k, largest=False)
        
        # Generate synthetic samples
        synthetic_list = []
        for _ in range(n_synthetic):
            i = torch.randint(0, n_pos, (1,)).item()
            neighbor_idx = nn_indices[i, torch.randint(0, k, (1,)).item()]
            neighbor_emb = pos_emb[neighbor_idx]
            
            lam = torch.rand(1, device=z.device).item() * self.alpha
            synthetic = pos_emb[i] + lam * (neighbor_emb - pos_emb[i])
            synthetic_list.append(synthetic)
        
        synthetic_emb = torch.stack(synthetic_list)
        
        z_new = torch.cat([z, synthetic_emb], dim=0)
        y_new = torch.cat([y, torch.ones(n_synthetic, device=y.device)], dim=0)
        
        return z_new, y_new

# Global SMOTE sampler instance
SMOTE_SAMPLER = SMOTESampler(
    k_neighbors=CONFIG['smote_k'],
    alpha=CONFIG['smote_alpha'],
    random_state=SEED
)

# =========================================
# GNN MODELS
# =========================================
class GCN(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, y=None, use_smote=False):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), self.dropout, self.training)
        
        if use_smote and y is not None and self.training:
            x, y = SMOTE_SAMPLER.fit_resample_latent(x, y)
        
        logits = self.fc(x).view(-1)
        return logits, y, x


class GAT(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, y=None, use_smote=False):
        x = F.dropout(F.elu(self.conv1(x, edge_index)), self.dropout, self.training)
        x = F.dropout(F.elu(self.conv2(x, edge_index)), self.dropout, self.training)
        
        if use_smote and y is not None and self.training:
            x, y = SMOTE_SAMPLER.fit_resample_latent(x, y)
        
        logits = self.fc(x).view(-1)
        return logits, y, x

class GraphSAGE(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, y=None, use_smote=False):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), self.dropout, self.training)
        
        if use_smote and y is not None and self.training:
            x, y = SMOTE_SAMPLER.fit_resample_latent(x, y)
        
        logits = self.fc(x).view(-1)
        return logits, y, x

class MLP(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None, y=None, use_smote=False):
        x = self.encoder(x)
        
        if use_smote and y is not None and self.training:
            x, y = SMOTE_SAMPLER.fit_resample_latent(x, y)
        
        logits = self.fc(x).view(-1)
        return logits, y, x


# ======================
# METRICS
# ======================
def calculate_metrics(y_true, y_pred, y_scores):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
        'PR-AUC': average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

# ==================================
# TRAINING AND EVALUATION
# ==================================
def train_gnn(model, X, y, full_edge_index, train_mask, val_mask, config):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    
    N = X.size(0)
    best_auc, best_state, patience = 0, None, 0
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        # Use FULL graph for message passing
        logits, targets, _ = model(X, full_edge_index, y, use_smote=config['use_smote'])
        
        num_syn = logits.shape[0] - N
        train_indices = torch.nonzero(train_mask).squeeze()
        
        if num_syn > 0:
            active_idx = torch.cat([train_indices, torch.arange(N, N + num_syn, device=DEVICE)])
        else:
            active_idx = train_indices
        
        # Only compute loss on TRAIN nodes
        loss = criterion(logits[active_idx], targets[active_idx])
        loss.backward()
        optimizer.step()
        
        # Validation using FULL graph
        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(X, full_edge_index, use_smote=False)
            probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
            true = y[val_mask].cpu().numpy()
            auc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else 0.5
        
        if auc > best_auc:
            best_auc, best_state, patience = auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= config['patience']:
                break
    
    model.load_state_dict(best_state)
    return model


def eval_gnn(model, X, y, full_edge_index, test_mask):
    
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X, full_edge_index, use_smote=False)
        probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
    y_true = y[test_mask].cpu().numpy()
    return calculate_metrics(y_true, (probs > 0.5).astype(int), probs)


def train_eval_ml(X_train, y_train, X_test, y_test, model_fn, use_smote=True):

    if use_smote:
        X_resampled, y_resampled = SMOTE_SAMPLER.fit_resample_numpy(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train
    
    model = model_fn()
    model.fit(X_resampled, y_resampled)
    
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
    
    return calculate_metrics(y_test, y_pred, y_scores)


# ================================
# CROSS-VALIDATION RUNNERS
# ================================
def run_gnn_cv(X_raw, y_np, full_edge_index, train_val_idx, model_class, name, config):
    
    print(f"  {name} (Latent SMOTE, Full Graph)...", end=" ")
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=SEED)
    N = len(y_np)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_idx, y_np[train_val_idx])):
        g_train, g_val = train_val_idx[train_idx], train_val_idx[val_idx]
        
        # Normalize features (leak-free)
        X = torch.tensor(normalize_features(X_raw, g_train, g_val), dtype=torch.float, device=DEVICE)
        y = torch.tensor(y_np, dtype=torch.float, device=DEVICE)
        
        # Create masks for label access only
        train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        train_mask[g_train], val_mask[g_val] = True, True
        
        model = model_class(X.shape[1], config['hidden_dim'], config['dropout']).to(DEVICE)
        model = train_gnn(model, X, y, full_edge_index, train_mask, val_mask, config)
        results.append(eval_gnn(model, X, y, full_edge_index, val_mask))
    
    mean_res = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std_res = {k: np.std([r[k] for r in results]) for k in results[0]}
    print(f"AUC: {mean_res['ROC-AUC']:.4f} +/- {std_res['ROC-AUC']:.4f}, F1: {mean_res['F1']:.4f}")
    return {'mean': mean_res, 'std': std_res}

def run_ml_cv(X_raw, y_np, train_val_idx, model_fn, name, config):
    '''Run cross-validation for ML with Input SMOTE'''
    print(f"  {name} (Input SMOTE)...", end=" ")
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=SEED)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_idx, y_np[train_val_idx])):
        g_train, g_val = train_val_idx[train_idx], train_val_idx[val_idx]
        X = normalize_features(X_raw, g_train, g_val)
        
        print(f"\n    Fold {fold + 1}/{config['n_folds']}:", end=" ")
        metrics = train_eval_ml(
            X[g_train], y_np[g_train], 
            X[g_val], y_np[g_val], 
            model_fn,
            use_smote=config['use_smote']
        )
        results.append(metrics)
    
    mean_res = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std_res = {k: np.std([r[k] for r in results]) for k in results[0]}
    print(f"\n  {name} Summary - AUC: {mean_res['ROC-AUC']:.4f} +/- {std_res['ROC-AUC']:.4f}, F1: {mean_res['F1']:.4f}")
    return {'mean': mean_res, 'std': std_res}

# ================================
# TEST RUN
# ================================
def run_gnn_test(X_raw, y_np, full_edge_index, train_val_idx, test_idx, model_class, config):
    
    N = len(y_np)
    
    # Normalize features refraining leakage
    X = torch.tensor(normalize_features(X_raw, train_val_idx, test_idx), dtype=torch.float, device=DEVICE)
    y = torch.tensor(y_np, dtype=torch.float, device=DEVICE)
    
    # Split train_val into train and val for early stopping
    ft_idx, fv_idx = train_test_split(train_val_idx, test_size=0.1, stratify=y_np[train_val_idx], random_state=SEED)
    
    # Create masks for label access only
    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    train_mask[ft_idx], val_mask[fv_idx], test_mask[test_idx] = True, True, True
    
    model = model_class(X.shape[1], config['hidden_dim'], config['dropout']).to(DEVICE)
    model = train_gnn(model, X, y, full_edge_index, train_mask, val_mask, config)
    return eval_gnn(model, X, y, full_edge_index, test_mask)


def run_ml_test(X_raw, y_np, train_val_idx, test_idx, model_fn, config):
    
    X = normalize_features(X_raw, train_val_idx, test_idx)
    print(f"\n    Test set SMOTE:", end=" ")
    return train_eval_ml(
        X[train_val_idx], y_np[train_val_idx], 
        X[test_idx], y_np[test_idx], 
        model_fn,
        use_smote=config['use_smote']
    )

# ========================================
# VISUALIZATION
# ========================================
def create_combined_heatmap(cv_results, test_results, model_names, gnn_names, ml_names):

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'MCC']
    
    # Prepare data with SMOTE type labels
    combined_data = []
    combined_index = []
    
    # Add CV results
    for m in model_names:
        smote_type = "Latent" if m in gnn_names else "Input"
        combined_data.append([cv_results[m]['mean'][metric] for metric in metrics])
        combined_index.append(f"{m} ({smote_type}) - CV")
    
    # Add Test results
    for m in model_names:
        smote_type = "Latent" if m in gnn_names else "Input"
        combined_data.append([test_results[m][metric] for metric in metrics])
        combined_index.append(f"{m} ({smote_type}) - Test")
    
    combined_df = pd.DataFrame(combined_data, index=combined_index, columns=metrics)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 14))
    
    sns.heatmap(combined_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                vmin=0, vmax=1, linewidths=0.5)
    
    # Add separator line between CV and Test results
    ax.axhline(y=len(model_names), color='black', linewidth=3)
    
    ax.set_title('Model Comparison: CV vs Test\n(GNN: Full Graph + Latent SMOTE, ML: Input SMOTE)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/comparison_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR}/plots/comparison_combined.png")
    
    return combined_df

# ===========================
# MAIN
# ============================
def main():
    print("=" * 70)
    print("MODEL COMPARISON: GNN vs ML")
    print("=" * 70)
    print("GNN Models: Full Graph (transductive) + Latent SMOTE")
    print("ML Models: Input SMOTE (in feature space)")
    print("=" * 70)
    
    # Load data
    X_raw, y_np, full_edge_index = load_data()
    
    # Split indices (labels only, not edges)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y_np)), test_size=0.2, stratify=y_np, random_state=SEED
    )
    
    # Verify no overlap
    assert len(set(train_val_idx) & set(test_idx)) == 0, "Data leak detected!"
    
    print(f"\nTrain/Val: {len(train_val_idx)}, Test: {len(test_idx)}")
    print(f"Full graph edges used for all nodes: {full_edge_index.shape[1]//2}")
    print(f"SMOTE enabled: {CONFIG['use_smote']}")
    print(f"SMOTE k-neighbors: {CONFIG['smote_k']}, alpha: {CONFIG['smote_alpha']}")
    
    # Define models
    gnn_models = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': GraphSAGE, 'MLP': MLP}
    ml_models = {
        'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        'SVM': lambda: SVC(kernel='rbf', probability=True, random_state=SEED),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'GradientBoost': lambda: GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    }
    
    gnn_names = list(gnn_models.keys())
    ml_names = list(ml_models.keys())
    model_names = gnn_names + ml_names
    
    cv_results, test_results = {}, {}
    
    # Cross-validation
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)
    
    print("\n--- GNN Models (Full Graph + Latent SMOTE) ---")
    for name, model_class in gnn_models.items():
        cv_results[name] = run_gnn_cv(X_raw, y_np, full_edge_index, train_val_idx, model_class, name, CONFIG)
    
    print("\n--- ML Models (Input SMOTE) ---")
    for name, model_fn in ml_models.items():
        cv_results[name] = run_ml_cv(X_raw, y_np, train_val_idx, model_fn, name, CONFIG)
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)
    
    print("\n--- GNN Models (Full Graph + Latent SMOTE) ---")
    for name, model_class in gnn_models.items():
        print(f"  {name}...", end=" ")
        test_results[name] = run_gnn_test(X_raw, y_np, full_edge_index, train_val_idx, test_idx, model_class, CONFIG)
        r = test_results[name]
        print(f"AUC: {r['ROC-AUC']:.4f}, F1: {r['F1']:.4f}, P: {r['Precision']:.4f}, R: {r['Recall']:.4f}")
    
    print("\n--- ML Models (Input SMOTE) ---")
    for name, model_fn in ml_models.items():
        print(f"  {name}...", end=" ")
        test_results[name] = run_ml_test(X_raw, y_np, train_val_idx, test_idx, model_fn, CONFIG)
        r = test_results[name]
        print(f"\n  {name} - AUC: {r['ROC-AUC']:.4f}, F1: {r['F1']:.4f}, P: {r['Precision']:.4f}, R: {r['Recall']:.4f}")
    
    # Save results and create visualization
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Create combined heatmap
    combined_df = create_combined_heatmap(cv_results, test_results, model_names, gnn_names, ml_names)
    
    # Save metrics to CSV
    combined_df.to_csv(f'{OUTPUT_DIR}/metrics/comparison_results.csv')
    print(f"Saved: {OUTPUT_DIR}/metrics/comparison_results.csv")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (Test Set)")
    print("=" * 80)
    print("-" * 95)
    print(f"{'Model':<15} {'SMOTE Type':<12} {'ROC-AUC':<10} {'PR-AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
    print("-" * 95)
    
    for m in model_names:
        r = test_results[m]
        smote_type = "Latent" if m in gnn_names else "Input"
        print(f"{m:<15} {smote_type:<12} {r['ROC-AUC']:<10.4f} {r['PR-AUC']:<10.4f} {r['Precision']:<10.4f} "
              f"{r['Recall']:<10.4f} {r['F1']:<10.4f} {r['MCC']:<10.4f}")
    
    print("\n" + "=" * 80)
    print("BEST MODELS BY METRIC:")
    print("=" * 80)
    for metric in ['ROC-AUC', 'PR-AUC', 'F1', 'MCC', 'Recall', 'Precision']:
        best = max(model_names, key=lambda x: test_results[x][metric])
        smote_type = "Latent SMOTE" if best in gnn_names else "Input SMOTE"
        print(f"   {metric:<12}: {best:<15} ({smote_type}) = {test_results[best][metric]:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()