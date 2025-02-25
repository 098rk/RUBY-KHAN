import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv(r"D:\ProjectFiles\Geeks\saptial\Parent_Visium_Human_OvarianCancer_analysis\analysis\umap\2_components\projection.csv")
print("Dataset Head:\n", df.head())

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Check for necessary columns for significant genes analysis
if {'log2FoldChange', 'padj'}.issubset(df.columns):
    # Identifying significant genes
    sig_threshold = 0.05
    df['Significant'] = (df['padj'] < sig_threshold) & (abs(df['log2FoldChange']) > 1)
    print(f"\nNumber of significant genes: {df['Significant'].sum()}")

    # Volcano Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='log2FoldChange', y=-np.log10(df['padj']), hue='Significant', alpha=0.7)
    plt.axhline(-np.log10(sig_threshold), linestyle='--', color='red', label='p = 0.05')
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10 Adjusted p-value")
    plt.title("Volcano Plot")
    plt.legend()
    plt.show()

# UMAP Plot using 'Barcode', 'UMAP-1', and 'UMAP-2'
plt.figure(figsize=(10, 8))

# Check if 'Significant' column exists and plot accordingly
if 'Significant' in df.columns:
    sns.scatterplot(data=df, x='UMAP-1', y='UMAP-2', hue='Significant', alpha=0.7)
else:
    sns.scatterplot(data=df, x='UMAP-1', y='UMAP-2', alpha=0.7)

plt.title("UMAP Plot")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

# Hierarchical Clustering Heatmap (requires 'gene' column)
if 'gene' in df.columns:
    df_sorted = df.sort_values('padj').head(50)  # Top 50 most significant genes
    heatmap_data = df_sorted.pivot(index='gene', columns='log2FoldChange', values='padj')
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True)
    plt.title("Hierarchical Clustering Heatmap")
    plt.show()

# PCA Analysis (requires 'gene' column)
if 'gene' in df.columns:
    numeric_data = df.drop(columns=['gene']).dropna()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Plot")
    plt.show()

# T-tests Between Clusters (if applicable)
if 'cluster' in df.columns:
    cluster_groups = df.groupby('cluster')['log2FoldChange']
    clusters = list(cluster_groups.groups.keys())
    if len(clusters) >= 2:
        t_stat, p_val = ttest_ind(cluster_groups.get_group(clusters[0]), cluster_groups.get_group(clusters[1]))
        print(f"\nT-test Between Clusters {clusters[0]} and {clusters[1]}: p-value = {p_val}")

print("\nAnalysis Complete.")



# Load the dataset
df = pd.read_csv(r"D:\ProjectFiles\Geeks\saptial\Parent_Visium_Human_OvarianCancer_analysis\analysis\tsne\2_components\projection.csv")
print("Dataset Head:\n", df.head())

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# t-SNE Plot using 'Barcode', 'TSNE-1', and 'TSNE-2'
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', alpha=0.7)
plt.title("t-SNE Plot")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.show()

print("\nAnalysis Complete.")

# Assuming you have a 'Significant' or 'Cluster' column to color points
if 'Significant' in df.columns:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', hue='Significant', alpha=0.7)
    plt.title("t-SNE Plot with Significance")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend()
    plt.show()
else:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', alpha=0.7)
    plt.title("t-SNE Plot")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.show()

# Step 1: Load the dataset
df = pd.read_csv(r"D:\ProjectFiles\Geeks\saptial\Parent_Visium_Human_OvarianCancer_analysis\analysis\pca\10_components\dispersion.csv")  # Update with your file path

# Step 2: Display basic information
print("Dataset Head:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

# Step 3: Summary statistics of Normalized.Dispersion
print("\nSummary Statistics for Normalized.Dispersion:\n", df['Normalized.Dispersion'].describe())

# Step 4: Visualizations

# Histogram of Normalized.Dispersion
plt.figure(figsize=(10, 6))
sns.histplot(df['Normalized.Dispersion'], bins=30, kde=True)
plt.title('Distribution of Normalized.Dispersion')
plt.xlabel('Normalized Dispersion')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Boxplot of Normalized.Dispersion
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Normalized.Dispersion'])
plt.title('Boxplot of Normalized.Dispersion')
plt.xlabel('Normalized Dispersion')
plt.grid()
plt.show()

# Step 5: Outlier Detection
# Define threshold for outliers (e.g., 1.5 * IQR)
Q1 = df['Normalized.Dispersion'].quantile(0.25)
Q3 = df['Normalized.Dispersion'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Normalized.Dispersion'] < lower_bound) | (df['Normalized.Dispersion'] > upper_bound)]
print("\nOutliers Detected:\n", outliers)

# Step 6: Correlation (if applicable)
# If you have other numerical features, you can analyze correlation.
# Uncomment and modify the following line as needed.
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

print("\nAnalysis Complete.")


# Load the dataset
df = pd.read_csv(r"D:\ProjectFiles\Geeks\saptial\Parent_Visium_Human_OvarianCancer_analysis\analysis\pca\10_components\projection.csv")

# Display the first few rows of the dataset
print("Dataset Head:\n", df.head())

# Display the shape of the DataFrame
print("\nDataset Shape:", df.shape)

# Summary statistics of the principal components (excluding the 'Barcode' column)
print("\nSummary Statistics:\n", df.describe())

# Visualizing the distribution of each principal component
plt.figure(figsize=(12, 8))
for i in range(1, 11):  # For PC-1 to PC-10
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[f'PC-{i}'], palette="Set3", hue=None)  # Changed to y to avoid warnings
    plt.title(f'Distribution of PC-{i}')
    plt.xlabel('Value')  # Changed from ylabel to xlabel for horizontal boxplots
plt.tight_layout()
plt.show()

# Visualizing the pairwise relationships between principal components
sns.pairplot(df.iloc[:, 1:11], diag_kind='kde', markers='o')
plt.suptitle("Pairwise Relationships Between Principal Components", y=1.02)
plt.show()

# Optional: PCA plot (if you want to visualize the first two principal components)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PC-1', y='PC-2', alpha=0.7)
plt.title("PCA Plot (PC-1 vs PC-2)")
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.show()

print("\nAnalysis Complete.")

# Load the dataset
df = pd.read_csv(r"D:\ProjectFiles\Geeks\saptial\Parent_Visium_Human_OvarianCancer_analysis\analysis\pca\10_components\components.csv")

# Display the first few rows of the dataset
print("Dataset Head:\n", df.head())

# Display the shape of the DataFrame
print("\nDataset Shape:", df.shape)

# Summary statistics of gene expression data (excluding the first column)
print("\nSummary Statistics:\n", df.describe())

# Visualizing the distribution of gene expression values
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.iloc[:, 1:], palette="Set3")
plt.title("Boxplot of Gene Expression Levels")
plt.xlabel("Genes")
plt.ylabel("Expression Levels")
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.show()

# Visualizing correlations between genes (if necessary)
correlation_matrix = df.iloc[:, 1:].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Gene Expression Correlation Matrix")
plt.show()

# Optional: Performing PCA for dimensionality reduction and visualization
from sklearn.decomposition import PCA

# Perform PCA on the gene expression data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.iloc[:, 1:])  # Skip the 'PC' column

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])

# Plot PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', alpha=0.7)
plt.title("PCA of Gene Expression Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

print("\nAnalysis Complete.")

