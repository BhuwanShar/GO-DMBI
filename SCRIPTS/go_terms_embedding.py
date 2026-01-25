import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BertForPreTraining
from tqdm import tqdm
import os


nodes_path = "Data/raw/node.csv"
go_terms_path = "Data/raw/gene_go_terms_descriptions.csv"

# Load node and GO data
nodes_df = pd.read_csv(nodes_path)  # SYMBOL, degree, is_biomarker
go_df = pd.read_csv(go_terms_path)  # Gene, GO_ID, Aspect, GO_Term, Namespace

# Clean and prepare GO annotations
go_df.drop_duplicates(subset=["Gene", "GO_ID"], inplace=True)
symbols = nodes_df["SYMBOL"].unique()
filtered_go_df = go_df[go_df["Gene"].isin(symbols)]

def format_go_text(row):
    return f"id: {row['GO_ID']}; name: {row['GO_Term']}; namespace: {row['Namespace']}"

filtered_go_df = filtered_go_df.copy()
filtered_go_df["go_text"] = filtered_go_df.apply(format_go_text, axis=1)

gene_go_data = (
    filtered_go_df
    .groupby("Gene")
    .agg({
        "GO_ID": list,
        "go_text": list
    })
    .reset_index()
    .rename(columns={"Gene": "SYMBOL"})
)

merged_df = pd.merge(nodes_df, gene_go_data, on="SYMBOL", how="left")
merged_df = merged_df.dropna(subset=["GO_ID"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load fine-tuned BioBERT from local directory
biobert_dir = "models/bio_bert_go_tuned"
biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_dir)
biobert_model = AutoModel.from_pretrained(biobert_dir)
biobert_model.to(device)
biobert_model.eval()


go_term_embeddings = []
go_term_gene_names = []

for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="GO term embeddings"):
    gene = row["SYMBOL"]
    go_texts = row["go_text"]
    if not go_texts: continue
    go_term_sequence = " ".join(go_texts)
    tokenized_input = biobert_tokenizer(
        go_term_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = biobert_model(
            input_ids=tokenized_input["input_ids"].to(device),
            attention_mask=tokenized_input["attention_mask"].to(device)
        )
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    go_term_embeddings.append(embedding)
    go_term_gene_names.append(gene)

go_term_embeddings = np.array(go_term_embeddings)
go_term_embeddings_tensor = torch.tensor(go_term_embeddings)
print(f"Generated {len(go_term_embeddings)} GO term embeddings with shape {go_term_embeddings.shape}")


outdir = "Data/processed/function_level_go_terms_embeddings"
os.makedirs(outdir, exist_ok=True)

# Save GO term embeddings
torch.save(go_term_embeddings_tensor, f"{outdir}/gene_embeddings_go_terms.pt")
go_term_df = pd.DataFrame(go_term_embeddings, columns=[f"dim_{i}" for i in range(go_term_embeddings.shape[1])])
go_term_df.insert(0, "SYMBOL", go_term_gene_names)
go_term_df.to_csv(f"{outdir}/gene_embeddings_go_terms.csv", index=False)
