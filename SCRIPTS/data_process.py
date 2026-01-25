import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

# Load GeoKG embeddings

emb = np.load("DATA/GeOKG_Embeddings/GeOKG_50dim.npy")
prt_list = pickle.load(open(
    "DATA/GeOKG/src_data/GO/prt_list.pkl", "rb"
))

uniprot_to_idx = {up: i for i, up in enumerate(prt_list)}
D_geokg = emb.shape[1]
# Build symbol
map_df = pd.read_csv(
    "DATA/GeOKG/raw_data/string2uniprot.tsv",
    sep="\t", header=None
)
map_df.columns = ["From", "UniProt", "EntryName", "GeneNames", "ProteinNames"]

symbol_to_uniprot = {}
for _, r in map_df.iterrows():
    if pd.isna(r["GeneNames"]):
        continue
    for g in str(r["GeneNames"]).split():
        symbol_to_uniprot[g.upper()] = r["UniProt"]
# Load nodes and map 
nodes = pd.read_csv(
    "DATA/processed_breast/processed_nodes_breast.csv"
)

nodes["SYMBOL"] = nodes["SYMBOL"].str.upper()
nodes["UniProt"] = nodes["SYMBOL"].map(symbol_to_uniprot)
nodes["emb_idx"] = nodes["UniProt"].map(uniprot_to_idx)

print("Total nodes:", len(nodes))
print("Nodes with GeoKG:", nodes["emb_idx"].notna().sum())

# remove nodes without geokg embeddings
nodes = nodes[nodes["emb_idx"].notna()].reset_index(drop=True)
print("Nodes after GeoKG filtering:", len(nodes))

# Build feature matrix
X_geokg = np.vstack([
    emb[int(idx)] for idx in nodes["emb_idx"]
]).astype(np.float32)

np.save("breast_node_features_geokg.npy", X_geokg)

#Load go terms embedding and align

go_term_emb = pd.read_csv(
    "DATA/processed_breast/function_level_go_terms_emb_breast/go_terms_embeddings_breast.csv"
)

go_term_emb["SYMBOL"] = go_term_emb["SYMBOL"].str.upper()

go_cols = go_term_emb.columns.drop("SYMBOL")
D_go = len(go_cols)


symbol_to_go = {
    r["SYMBOL"]: r[go_cols].values.astype(np.float32)
    for _, r in go_term_emb.iterrows()
}
# Remove nodes without gookg embedding
has_go = nodes["SYMBOL"].isin(symbol_to_go)
nodes = nodes[has_go].reset_index(drop=True)

print("Nodes after GO filtering:", len(nodes))
# Build clean go_term feature matrix
X_go = np.vstack([
    symbol_to_go[sym] for sym in nodes["SYMBOL"]
]).astype(np.float32)

np.save("breast_node_features_go_terms.npy", X_go)


nodes.to_csv("breast_nodes_with_uniprot_and_embidx.csv", index=False)

# Curate edge 
edges = pd.read_csv(
    "DATA/Raw/BrestCancer_microarray/breast_edges.csv"
)

valid_symbols = set(nodes["SYMBOL"])
edges = edges[
    edges["source"].str.upper().isin(valid_symbols) &
    edges["target"].str.upper().isin(valid_symbols)
].copy()

edges["source"] = edges["source"].str.upper()
edges["target"] = edges["target"].str.upper()

edges.to_csv("breast_edges_curated.csv", index=False)

print("Final edges:", len(edges))

