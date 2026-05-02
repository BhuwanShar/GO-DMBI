setwd("D:\\DESSERTATION\\Biomarker_project\\FINAL_work\\DATA")
##########################################################
# DEG ANALYSIS # 
##########################################################
# Required libraries
library(GEOquery)
library(limma)
library(ggplot2)
library(pheatmap)
library(Biobase)
library(ggrepel)
# Download data
data <- getGEO("GSE16011", AnnotGPL = TRUE)

# Extract expression and pheno data
exprs_mat <- exprs(data[[1]])
pheno <- pData(data[[1]])

# Define treatment vs control groups
# Adjust based on your specific column name and keywords
# For example, assume column "title" or "characteristics_ch1"
# Customize this to match your dataset
group_column <- "title"  # or "characteristics_ch1", etc.

# Define what counts as control / treatment
treatment_keywords <- c("glioma")#c("Tumor_fresh-frozen_GBM", "glioma", "tumor")
control_keywords <- c("control")

# Create regex pattern
treatment_pattern <- paste(treatment_keywords, collapse = "|")
control_pattern <- paste(control_keywords, collapse = "|")

# Classify samples
group <- ifelse(grepl(control_pattern, pheno[[group_column]], ignore.case = TRUE), 
                "Control",
                ifelse(grepl(treatment_pattern, pheno[[group_column]], ignore.case = TRUE),
                       "Treatment", NA))
# Add classification to pheno
pheno$group <- group
# Show counts
table(pheno$group, useNA = "ifany")
# Remove unclassified samples
keep_samples <- !is.na(group)
group <- group[keep_samples]
pheno <- pheno[keep_samples, ]
exprs_mat <- exprs_mat[, keep_samples]
# Now check here what the scenario is of the new box plot
boxplot(exprs_mat, las = 2, outline = FALSE, main = "Expression Boxplot", ylab = "Expression")
# normalization required
exprs_mat <-normalizeQuantiles(exprs_mat)
pheno$group <- factor(group, levels = c("Control", "Treatment"))


# Build design matrix
design <- model.matrix(~ 0 + pheno$group)
colnames(design) <- levels(pheno$group)

# Fit model with limma
fit <- lmFit(exprs_mat, design)
contrast.matrix <- makeContrasts(Treatment - Control, levels = design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

# Get DEGs
deg_results <- topTable(fit2, adjust.method = "BH", number = Inf)
deg_filtered <- subset(deg_results, adj.P.Val < 0.05 & abs(logFC) > 2)

# Extract DEG expression matrix
sig_genes <- rownames(deg_filtered)
deg_exprs <- exprs_mat[sig_genes, ]

# Save DEG results to CSV
write.csv(deg_filtered, "DEG_results_GSE16011_limma.csv")

# Volcano Plot
deg_results$threshold <- with(deg_results, 
                              ifelse(adj.P.Val < 0.05 & abs(logFC) > 2, "Significant", "Not Significant"))

# Volcano Plot 
volcano_plot <- ggplot(deg_results, aes(x = logFC, y = -log10(adj.P.Val))) +
  geom_point(aes(color = threshold), alpha = 0.6) +
  scale_color_manual(values = c("Not Significant" = "grey", "Significant" = "red")) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "blue") +
  geom_hline(yintercept = -log10(0.05), linetype = "dotted", color = "black") +
  theme_minimal() +
  labs(title = "Volcano Plot: DEGs",
       x = "log2 Fold Change",
       y = "-log10 Adjusted P-Value")

# ggsave("volcano_plot_DEGs.png", width = 7, height = 5) to save
# Display the plot
print(volcano_plot)

# (Optional) Save the plot later if needed
# ggsave("volcano_plot_DEGs.png", plot = volcano_plot, width = 7, height = 5)


# Heatmap of DEGs
# Ensure column names match sample names
annotation_df <- data.frame(Group = pheno$group)
scaled_deg_exprs <- t(scale(t(deg_exprs)))
rownames(annotation_df) <- colnames(scaled_deg_exprs)

# Now plot
pheatmap(scaled_deg_exprs, 
         annotation_col = annotation_df,
         show_rownames = FALSE,
         fontsize_col = 8,
         main = "Heatmap of Significant DEGs")


# Save DEG expression matrix
write.csv(deg_exprs, "DEG_GSE16011_expression_matrix.csv")

# We will go for annotation later after we create the GRN for clean workflow and also annotiing only important genes

########################################################
####Since we have more dim of sig_genes perform pca#####
########################################################
# Input: deg_exprs (genes x samples)

# Transpose the expression matrix
# PCA expects rows = samples, columns = variables (genes)
'''exprs_t <- t(deg_exprs)

# Center the data (mean zero per gene)
exprs_centered <- scale(exprs_t, center = TRUE, scale = FALSE)

# Compute covariance matrix
cov_matrix <- cov(exprs_centered)

# Eigen decomposition
eig <- eigen(cov_matrix)

# Calculate principal components (PC scores)
pc_scores <- exprs_centered %*% eig$vectors  # Samples x PCs

# Determine how many PCs to retain (95% cumulative variance)
eig_vals <- eig$values
variance_explained <- eig_vals / sum(eig_vals)
cumulative_variance <- cumsum(variance_explained)

# Find number of PCs that explain at least 95% of variance
top_n_pcs <- which(cumulative_variance >= 0.95)[1]
cat("Number of PCs explaining ≥95% variance:", top_n_pcs, "\n")

# Subset PC scores matrix to top PCs
pc_matrix <- pc_scores[, 1:top_n_pcs]

# Get loadings (gene contributions to PCs)
loadings <- eig$vectors  # Genes x PCs
rownames(loadings) <- colnames(exprs_centered)  # Gene names
colnames(loadings) <- paste0("PC", 1:ncol(loadings))

# Keep loadings for top PCs only
gene_loadings_top <- loadings[, 1:top_n_pcs]

# Score total gene contribution across selected PCs
gene_contributions <- rowSums(abs(gene_loadings_top))

# Filter genes — remove bottom 5% least contributing genes
cutoff <- quantile(gene_contributions, probs = 0.05)  # Bottom 5%
selected_genes <- names(gene_contributions[gene_contributions > cutoff])

# Subset the original DEG expression matrix
pca_filtered_exprs <- deg_exprs[selected_genes, ]

# Done! WE now have a PCA-refined gene expression matrix with gene IDs retained
cat("Original genes:", nrow(deg_exprs), "\n")
cat("Genes retained after PCA filtering:", length(selected_genes), "\n")
write.csv(pca_filtered_exprs , "pcafilt_DEG_GSE16011_exp_matrix.csv")'''
