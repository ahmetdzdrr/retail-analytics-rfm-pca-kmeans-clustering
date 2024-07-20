# Import necessary functions from utils module
from utils import *

# Load the dataset
df = pd.read_excel("data/online_retail_II.xlsx", engine='openpyxl')

# Display the first few rows of the dataset
df.head()

# Get a summary of the dataframe
df.info()

# Provide an overview of the dataset (e.g., basic statistics, types of data)
data_overview(df)

# Check for missing values in the dataset
missing_values(df)

# Analyze the frequency of stock levels in the dataset
stock_frequency(df)

# Analyze the frequency of item descriptions in the dataset
description_frequency(df)

# Process dates in the dataset to standardize formats
df = date_process(df)

# Clean and process the data, separating sales and cancelled transactions
sales_df, cancelled_df = clean_process(df)

# Create a 3D map to visualize sales by country
country_map_3d(sales_df)

# Plot top 5 countries by quantity for sales data
plot_top_5_countries(sales_df, country_col="Country", target_col="Quantity")

# Plot top 5 countries by quantity for cancelled transactions
plot_top_5_countries(cancelled_df, country_col="Country", target_col="Quantity", is_cancelled=True)

# Perform RFM (Recency, Frequency, Monetary) segmentation on the sales data
rfm_df = rfm_segmentation(sales_df)

# Display the first few rows of the RFM dataframe
rfm_df.head()

# Compute RFM scores
rfm_df = rfm_score(rfm_df)

# Display the updated RFM dataframe with scores
rfm_df.head()

# Perform RFM score clustering and obtain segmentation map
rfm_score_df, seg_map = rfm_score_clustering(rfm_df)

# Display the first few rows of the RFM score dataframe
rfm_score_df.head()

# Analyze RFM segments
rfm_segment_analysis(rfm_score_df)

# Visualize RFM segments using a tree map
tree_map_segment(rfm_score_df, seg_map)

# Segment customers based on RFM scores
segment_customers(rfm_score_df)

# Display descriptive statistics for each segment
segment_desc_statistics(rfm_score_df)

# Process additional features based on the last purchase data
new_df = last_purchase(sales_df)

# Add features related to transactions and purchases
new_df = transaction_and_purchased(sales_df, new_df)

# Add features related to total and average spend
new_df = total_and_average_spend(sales_df, new_df)

# Add features related to product diversity
new_df = product_diversity(sales_df, new_df)

# Add behavior-based features to the dataset
new_df = behavior_features(sales_df, new_df)

# Add geographic-based features to the dataset
new_df = geographic_features(sales_df, new_df)

# Add seasonality trends to the dataset
new_df = seasonality_trends(sales_df, new_df)

# Perform correlation analysis on the new features
correlation_analysis(new_df)

# Scale the new features for further analysis
new_df_scaled = scale_method(new_df)

# Perform dimensionality reduction using PCA
dimensionality_reduction(new_df_scaled)

# Highlight top 3 features in PCA components
customer_data_pca, _ = highlight_top3_pca(new_df_scaled)

# Perform KMeans clustering on the PCA-transformed data
kmeans_clustering(customer_data_pca)

# Perform silhouette analysis to determine the optimal number of clusters
silhouette_analysis(customer_data_pca, 2, 11, figsize=(20, 50))

# Assign cluster labels to the original data
new_df, customer_data_pca = clustering_feature(new_df, customer_data_pca)

# Visualize the clustering results
clustering_visualization(customer_data_pca)

# Evaluate clustering performance using various metrics
eval_metric(customer_data_pca)

# Create radar charts for cluster centroids
radar_map(new_df)

# Re-visualize RFM segments with a tree map
tree_map_segment(rfm_score_df, seg_map)

# Plot histograms for each feature segmented by cluster
segmentation_features(new_df)
