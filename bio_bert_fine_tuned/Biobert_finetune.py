import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# Load and clean dataset
df = pd.read_csv(".../gene_go_input_texts.csv")
df = df.dropna(subset=["GO_Description_Text"])
df = df.rename(columns={"GO_Description_Text": "text"})

# Train-validation split (80-20)
train_texts, val_texts = train_test_split(df[["text"]], test_size=0.2, random_state=42)

# Convert to Huggingface Datasets
train_dataset = Dataset.from_pandas(train_texts)
val_dataset = Dataset.from_pandas(val_texts)
datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Load tokenizer and model
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

# Tokenize datasets
tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# DataLoaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=16, shuffle=False, collate_fn=data_collator)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training parameters
epochs = 10
train_losses = []
val_losses = []

# Training loop with validation
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1} Training")
        loop.set_postfix(loss=loss.item())
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Plot Loss curves
plt.figure(figsize=(8,6))
plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.savefig("./model/perplexity_curve.png")  # Save the figure





from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "[BP] anatomical structure morphogenesis; anterior/posterior pattern specification; embryonic skeletal system development; embryonic skeletal system morphogenesis; endothelial cell differentiation; positive regulation of transcription by RNA polymerase II; regulation of DNA-templated transcription; regulation of transcription by RNA polymerase II [CC] chromatin; cytosol; fibrillar center; nucleoplasm; nucleus [MF] DNA binding; DNA-binding transcription activator activity, RNA polymerase II-specific; DNA-binding transcription factor activity; DNA-binding transcription factor activity, RNA polymerase II-specific; RNA polymerase II cis-regulatory region sequence-specific DNA binding; protein binding; sequence-specific double-stranded DNA binding"

tokens = tokenizer.tokenize(text)
print(f"Number of tokens: {len(tokens)}") 
print(f"Tokens: {tokens}")

from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load locally fine-tuned model
model_path = ".../go_term_bert/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.to(device)
model.eval()
