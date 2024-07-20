# Customer Segmentation and Sales Analysis

This repository contains a comprehensive analysis and segmentation of customer data from an online retail dataset. It involves various stages of data preprocessing, feature engineering, clustering, and visualization to understand customer behavior and sales trends.

## Project Overview

This project focuses on:
- Data preprocessing and cleaning
- Feature engineering for customer behavior and sales analysis
- Customer segmentation using KMeans clustering
- Evaluation and visualization of clusters
- RFM (Recency, Frequency, Monetary) segmentation for further analysis

## Dataset

The dataset used in this project is the Online Retail II dataset, which includes transaction data for a UK-based online retail company. The data can be found in the `data` folder.

## Files

- `utils.py`: Contains utility functions for data preprocessing, feature engineering, and visualization.
- `main.py`: The main script for executing the data analysis, clustering, and visualization.
- `requirements.txt`: Lists the required Python packages and their versions for running the scripts.

## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name


2. **Create and Activate a Virtual Environment:**

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`


3. **Install Dependencies:**

    ```bash
    pip install requirements.txt


## Usage

1. **Run the main.py Script:**

    ```
    python main.py

This script will process the data, perform clustering, and generate visualizations.

2. **Functions and Features:**

> **Data Preprocessing**:
> - `date_process(df):` Processes and cleans date-related columns.
> - `clean_process(df):` Cleans and processes the raw dataset.

> **Feature Engineering**:
> - `transaction_and_purchased(df, new_df):` Computes transaction and purchase-related features.
> - `total_and_average_spend(df, new_df):` Calculates total and average spending per customer.
> - `product_diversity(df, new_df):` Measures product diversity per customer.
> - `behavior_features(df, new_df):` Extracts behavioral features such as shopping day and hour.
> - `geographic_features(df, new_df):` Adds geographic features like country of origin.
> - `seasonality_trends(df, new_df):` Analyzes seasonal trends in spending.
> - `transaction_and_purchased(df, new_df):` Computes transaction and purchase-related features.
> - `total_and_average_spend(df, new_df):` Calculates total and average spending per customer.
> - `product_diversity(df, new_df):` Measures product diversity per customer.
> - `behavior_features(df, new_df):` Extracts behavioral features such as shopping day and hour.
> - `geographic_features(df, new_df):` Adds geographic features like country of origin.
> - `seasonality_trends(df, new_df):` Analyzes seasonal trends in spending.

> **Clustering and Evaluation**:
> - `kmeans_clustering(customer_data_pca):` Performs KMeans clustering.
> - `silhouette_analysis(customer_data_pca, start_k, stop_k):` Analyzes silhouette scores for different values of k.
> - `clustering_feature(dataframe, customer_data_pca):` Adds clustering features to the dataset.
> - `eval_metric(customer_data_pca):` Evaluates clustering performance with various metrics.
> - `radar_map(dataframe):` Generates radar charts for cluster centroids.
> - `segmentation_features(dataframe):` Plots histograms of features segmented by clusters.


## Visualizations

The project includes several visualizations to understand customer segmentation and sales patterns, including:

- 3D country maps
- Top 5 countries by quantity
- Distribution of customers across clusters
- Radar charts for cluster centroids
- Histograms of features segmented by clusters

## Contribution

Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
