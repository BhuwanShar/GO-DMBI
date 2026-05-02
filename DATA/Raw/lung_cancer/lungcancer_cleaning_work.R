setwd("D:\\DESSERTATION\\Biomarker_project\\FINAL_work\\DATA\\lung_cancer")
data_1 <- read.delim("GSE19804.top.table.tsv")
data_2 <- read.delim("GSE31210.top.table.tsv")
deg_filtered_data1 <- subset(data_1, adj.P.Val < 0.05 & abs(logFC) > 1.5)
deg_filtered_data2 <- subset(data_2, adj.P.Val < 0.05 & abs(logFC) > 2)
dim(deg_filtered_data1)
dim(deg_filtered_data2)

merged_symbols <- union(deg_filtered_data12$Gene.symbol,deg_filtered_data1$Gene.symbol)
length(merged_symbols)
write.csv(merged_symbols, "total_genes.csv")

setwd("D:\\DESSERTATION\\Biomarker_project\\FINAL_work\\DATA\\lung_cancer\\Biomarkers")
biom_data_gene <- read.delim("Gene.tsv")
biom_data_protein <- read.delim("Protein.tsv")
gene_marker_ids <- biom_data_gene$biomarkername # Both files have same content
protein_marker_ids <- biom_data_protein$biomarkername
gene_hits     <- gene_marker_ids[grepl("^[A-Z0-9-]+$", gene_marker_ids)]
protein_hits  <- protein_marker_ids[grepl("^[A-Z0-9-]+$", protein_marker_ids)]
gene_protein_hits <- union(gene_hits, protein_hits)
gene_protein_common <- intersect(gene_hits, protein_hits)
gene_hits              # likely genes from Gene.tsv
protein_hits           # likely genes from Protein.tsv
gene_protein_hits      # union (all unique likely gene IDs)
gene_protein_common
write.csv(protein_hits,"curated_biomarkerids.csv")

################################################################################
# Post PPI work
setwd("D:\\DESSERTATION\\Biomarker_project\\FINAL_work\\DATA\\lung_cancer")

library(igraph)


string_data <- read.delim("string_interactions_short_lung.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)

head(string_data)


g <- graph_from_data_frame(string_data[, c("X.node1", "node2")], directed = FALSE)


g_filtered <- delete_vertices(g, V(g)[degree(g) < 2])


print(g)
print(g_filtered)


cat("Removed", vcount(g) - vcount(g_filtered), "nodes with degree < 2\n")


filtered_edges <- as_data_frame(g_filtered, what = "edges")
write.table(filtered_edges, "filtered_string_edges.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
nodes_data <- read.csv("string_interactions_short.tsv default node.csv")
biom_in_graph <- intersect(nodes_data$name,gene_hits)
