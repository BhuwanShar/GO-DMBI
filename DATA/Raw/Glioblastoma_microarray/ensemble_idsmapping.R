setwd("D://DESSERTATION//Biomarker_project//FINAL_work//DATA//Glioblastoma_microarray")
data <- read.delim("GSE16011_GEO2R.tsv")
deg_genes <-subset(data, adj.P.Val < 0.05 & abs(logFC) > 2)
library(biomaRt)

library(biomaRt)

# Example input data
data_1 <- data.frame(
  ID = deg_genes$ID,
  Chr = deg_genes$Chr,
  Chr.Strand = deg_genes$Chr.Strand,
  Chr.From = deg_genes$Chr.From,
  Chr.To   = deg_genes$Chr.To
)

# Always make sure start < end
data$Start <- pmin(data_1$Chr.From, data$Chr.To)
data$End   <- pmax(data_1$Chr.From, data$Chr.To)

# Connect to Ensembl human genes
library(biomaRt)

ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")

# Query Ensembl by chromosomal region
mapped_list <- lapply(seq_len(nrow(data)), function(i) {
  getBM(
    attributes = c("ensembl_gene_id", "hgnc_symbol", "entrezgene_id",
                   "chromosome_name", "start_position", "end_position", "strand"),
    filters = c("chromosome_name", "start", "end"),
    values = list(data$Chr[i], data$Start[i], data$End[i]),
    mart = ensembl
  )
})

# Attach probe IDs back
for (i in seq_along(mapped_list)) {
  if (nrow(mapped_list[[i]]) > 0) {
    mapped_list[[i]]$ID <- data$ID[i]
  }
}

# Combine into one dataframe
mapped_genes <- do.call(rbind, mapped_list)

cleaned_genes <- mapped_genes %>%
  filter(!is.na(hgnc_symbol)) %>%        # remove NA gene symbols
  distinct(hgnc_symbol, .keep_all = TRUE)  # keep only first occurrence of each gene symbol

head(cleaned_genes)

cleaned_genes <- mapped_genes %>%
  filter(!is.na(hgnc_symbol)) %>%
  arrange(desc(!is.na(entrezgene_id))) %>%  # prioritize rows with valid Entrez ID
  distinct(hgnc_symbol, .keep_all = TRUE)





write.csv(cleaned_genes, "mapped_genes_gse11_updated.csv")
