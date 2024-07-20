import pandas as pd
import squarify
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot

import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tabulate import tabulate
from sklearn.cluster import KMeans
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)
pd.set_option('display.float_format', lambda x: '%.0f' % x)
sns.set_style('whitegrid')
palette = 'Set2'
colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']



def data_overview(dataframe):
    """
    Provides an overview of the given DataFrame including its shape, data types, and duplicated values.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame for which to provide the overview.

    Returns:
    None
    """
    
    # Print the shape of the dataset
    print(" SHAPE OF DATASET ".center(30, '-'))
    print('Rows:{}'.format(dataframe.shape[0]))
    print('Columns:{}'.format(dataframe.shape[1]))
    print()
    
    # Print the data types and their counts
    print(" DATA TYPES ".center(30, '-'))
    print(dataframe.dtypes.value_counts())
    print()
    
    # Print the number of duplicated values
    print(" DUPLICATED VALUES ".center(30, '-'))
    print(dataframe.duplicated().sum())


def missing_values(dataframe):
    """
    Analyzes and visualizes the missing values in the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to analyze for missing values.

    Returns:
    None
    """
    
    # Calculate the number of missing values in each column
    missing_data = dataframe.isnull().sum()
    
    # Calculate the percentage of missing values
    missing_percentage = (missing_data[missing_data > 0] / dataframe.shape[0]) * 100

    # Sort the missing percentages in ascending order
    missing_percentage.sort_values(ascending=True, inplace=True)

    # Plot the horizontal bar chart
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.barh(missing_percentage.index, missing_percentage, color='#ff6200')

    # Annotate the percentage values next to the bars
    for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
        ax.text(value + 0.5, i, f"{value:.2f}%", ha='left', va='center', fontweight='bold', fontsize=18, color='black')

    # Set the x-axis limit
    ax.set_xlim([0, 40])

    # Add title and x-axis label
    plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
    plt.xlabel('Percentages (%)', fontsize=16)
    
    # Display the plot
    plt.show()



def stock_frequency(dataframe):
    """
    Analyzes and visualizes the top 10 most frequent stock codes in the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to analyze for stock code frequency.

    Returns:
    None
    """
    
    # Calculate the percentage frequency of the top 10 most frequent stock codes
    top_10_stock_codes = dataframe['StockCode'].value_counts(normalize=True).head(10) * 100

    # Plotting the top 10 most frequent stock codes as a horizontal bar chart
    plt.figure(figsize=(12, 8))
    top_10_stock_codes.plot(kind='barh', color='#ff6200')

    # Adding the percentage frequency on the bars
    for index, value in enumerate(top_10_stock_codes):
        plt.text(value, index + 0.25, f'{value:.2f}%', fontsize=10)

    # Add title and axis labels
    plt.title('Top 10 Most Frequent Stock Codes')
    plt.xlabel('Percentage Frequency (%)')
    plt.ylabel('Stock Codes')

    # Invert y-axis to have the most frequent stock code on top
    plt.gca().invert_yaxis()
    
    # Display the plot
    plt.show()



def description_frequency(dataframe):
    """
    Analyzes and visualizes the top 10 most frequent product descriptions in the given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to analyze for product description frequency.

    Returns:
    None
    """
    
    # Calculate the frequency of each product description
    description_counts = dataframe['Description'].value_counts()

    # Get the top 10 most frequent descriptions
    top_30_descriptions = description_counts[:10]

    # Plotting the top 10 most frequent descriptions as a horizontal bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(top_30_descriptions.index[::-1], top_30_descriptions.values[::-1], color='#ff6200')

    # Adding labels and title
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Description')
    plt.title('Top 10 Most Frequent Descriptions')

    # Display the plot
    plt.show()



def date_process(dataframe):
    """
    Processes the input DataFrame by extracting year, month, and day from the 'InvoiceDate' column,
    and calculates the total price for each invoice.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing the columns 'InvoiceDate', 'Quantity', and 'Price'.

    Returns:
    pd.DataFrame: The input DataFrame with additional columns:
        - 'year': Extracted year from 'InvoiceDate'.
        - 'month': Extracted month from 'InvoiceDate'.
        - 'day': Extracted day from 'InvoiceDate'.
        - 'total_price': Calculated total price (Quantity * Price).
    """
    # Extract the year from 'InvoiceDate' and add it as a new column 'year'
    dataframe['year'] = dataframe['InvoiceDate'].dt.year
    
    # Extract the month from 'InvoiceDate' and add it as a new column 'month'
    dataframe['month'] = dataframe['InvoiceDate'].dt.month
    
    # Extract the day from 'InvoiceDate' and add it as a new column 'day'
    # Note: It should be 'day' instead of 'month'
    dataframe['day'] = dataframe['InvoiceDate'].dt.day
    
    # Calculate the total price by multiplying 'Quantity' with 'Price' and add it as a new column 'total_price'
    dataframe['total_price'] = dataframe['Quantity'] * dataframe['Price']

    return dataframe



def clean_process(dataframe):
    """
    Cleans the input DataFrame by removing duplicates, dropping rows with missing values, and separating 
    the data into sales and cancelled transactions based on the 'Invoice' column.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing the columns 'Invoice'.

    Returns:
    tuple: A tuple containing two DataFrames:
        - saled_dataframe (pd.DataFrame): DataFrame with sales data, excluding rows where 'Invoice' contains 'C'.
        - cancelled_dataframe (pd.DataFrame): DataFrame with cancelled data, including only rows where 'Invoice' contains 'C'.
    """
    # Remove duplicate rows, keeping the first occurrence
    dataframe = dataframe.drop_duplicates(keep="first")
    
    # Drop rows with missing values in any column
    dataframe = dataframe.dropna()
    
    # Filter out rows where 'Invoice' contains 'C' (cancelled transactions)
    saled_dataframe = dataframe[~dataframe.Invoice.str.contains('C', na=False)]
    
    # Filter rows where 'Invoice' contains 'C' (cancelled transactions)
    cancelled_dataframe = dataframe[dataframe.Invoice.str.contains('C', na=False)]
    
    # Print the shapes of the sales and cancelled DataFrames
    print(f"Sales data shape: {saled_dataframe.shape} \nCancelled data shape: {cancelled_dataframe.shape}")

    return saled_dataframe, cancelled_dataframe



def country_map_3d(dataframe):
    """
    Creates a 3D choropleth map displaying the number of orders by country using Plotly.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing columns 'Customer ID', 'Invoice', and 'Country'.

    Returns:
    None: Displays an interactive choropleth map.
    """
    # Group by 'Customer ID', 'Invoice', and 'Country', then count occurrences
    world_map = dataframe[['Customer ID', 'Invoice', 'Country']
                          ].groupby(['Customer ID', 'Invoice', 'Country']
                                    ).count().reset_index(drop=False)
    
    # Count the number of orders for each country
    countries = world_map['Country'].value_counts()
    
    # Define the data for the choropleth map
    data = dict(
        type='choropleth',
        locations=countries.index,  # Country names
        locationmode='country names',
        z=countries,  # Number of orders
        text=countries.index,  # Country names for hover text
        colorbar={'title': 'Orders'},  # Colorbar title
        colorscale='Viridis',  # Color scale
        reversescale=False
    )
    
    # Define the layout for the choropleth map
    layout = dict(
        title={
            'text': "Number of Orders by Countries",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        geo=dict(
            resolution=50,
            showocean=True,
            oceancolor="LightBlue",
            showland=True,
            landcolor="whitesmoke",
            showframe=True
        ),
        template='plotly_white',
        height=600,
        width=1000
    )
    
    # Create and display the choropleth map using Plotly
    choromap = go.Figure(data=[data], layout=layout)
    iplot(choromap, validate=False)



def plot_top_5_countries(dataframe, country_col, target_col, is_cancelled=False):
    """
    Plots the top 5 countries with the highest or lowest total quantity and total price based on the input DataFrame.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing the columns specified by `country_col` and `target_col`, and a 'total_price' column.
    country_col (str): The name of the column representing the country.
    target_col (str): The name of the column representing the quantity to be summed.
    is_cancelled (bool): If True, plots the top 5 countries with the lowest total quantity; otherwise, plots the top 5 countries with the highest total quantity.

    Returns:
    None: Displays two bar plots for total quantity and total price.
    """
    # Calculate the total quantity and total price by country
    total_quantity = dataframe.groupby(country_col)[target_col].sum().reset_index()
    total_price = dataframe.groupby(country_col)["total_price"].sum().reset_index()

    # Determine top 5 countries based on quantity (lowest or highest) and set the plot title
    if is_cancelled:
        top_5_countries = total_quantity.nsmallest(5, target_col)[country_col]
        title = 'Cancelled'
    else:
        top_5_countries = total_quantity.nlargest(5, target_col)[country_col]
        title = 'Sales'

    # Filter the data to include only the top 5 countries
    total_quantity_top_5 = total_quantity[total_quantity[country_col].isin(top_5_countries)]
    total_price_top_5 = total_price[total_price[country_col].isin(top_5_countries)]

    # Calculate the percentage of total quantity for each top country
    total_quantity_top_5['quantity_percentage'] = (total_quantity_top_5[target_col] / dataframe[target_col].sum()) * 100

    # Define colors for the plots
    colors = px.colors.qualitative.Vivid

    # Create bar plots for total quantity and total price
    fig_total_quantity = go.Figure()
    fig_total_price = go.Figure()

    # Format total price for text display
    total_price_text = [f"${val:.2f}" for val in total_price_top_5["total_price"]]

    # Add bar traces to the total quantity figure
    fig_total_quantity.add_trace(go.Bar(
        x=total_quantity_top_5[country_col],
        y=total_quantity_top_5[target_col],
        marker_color=colors[:5],
        text=[f"{val:.2f}%" for val in total_quantity_top_5['quantity_percentage']],
        textposition='auto'
    ))

    # Add bar traces to the total price figure
    fig_total_price.add_trace(go.Bar(
        x=total_price_top_5[country_col],
        y=total_price_top_5["total_price"],
        marker_color=colors[:5],
        text=total_price_text,
        textposition='auto'
    ))

    # Update the layout for total quantity figure
    fig_total_quantity.update_layout(
        title=f'({title}) The Highest 5 Total Quantity by Country',
        xaxis_title='Country',
        yaxis_title='Total Quantity',
        barmode='group'
    )

    # Update the layout for total price figure
    fig_total_price.update_layout(
        title=f'({title}) The Highest 5 Total Price by Country',
        xaxis_title='Country',
        yaxis_title='Total Price',
        barmode='group'
    )

    # Display the figures
    fig_total_quantity.show()
    fig_total_price.show()



def rfm_segmentation(dataframe):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for customer segmentation.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing customer transaction data with columns 
                              'Customer ID', 'InvoiceDate', 'Invoice', and 'total_price'.

    Returns:
    pd.DataFrame: DataFrame with RFM metrics for each customer.
    """

    # Calculate the date for performance measurement
    performans_date = dataframe["InvoiceDate"].max() + timedelta(days=2)
    
    # Aggregate the RFM metrics
    rfm_df = dataframe.groupby("Customer ID").agg(
        recency=('InvoiceDate', lambda x: (performans_date - x.max()).days),
        frequency=('Invoice', 'nunique'),
        monetary=('total_price', 'sum')
    )
    
    return rfm_df



def rfm_score(dataframe):
    """
    Calculates RFM (Recency, Frequency, Monetary) scores for each customer in the input DataFrame.
    The RFM score is used for customer segmentation based on their purchasing behavior.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing columns 'recency', 'frequency', and 'monetary' for each customer.

    Returns:
    pd.DataFrame: The input DataFrame with additional columns:
        - 'recency_score': Recency score based on quantiles (1 to 5).
        - 'frequency_score': Frequency score based on quantiles (1 to 5).
        - 'monetary_score': Monetary score based on quantiles (1 to 5).
        - 'RFM_SCORE': Concatenated score combining 'recency_score' and 'frequency_score'.
    """
    # Calculate the recency score by dividing 'recency' into 5 quantiles, assigning scores from 5 to 1 (lower values are better)
    dataframe["recency_score"] = pd.qcut(dataframe['recency'], 5, [5, 4, 3, 2, 1])
    
    # Calculate the frequency score by ranking 'frequency' and dividing into 5 quantiles, assigning scores from 1 to 5 (higher values are better)
    dataframe["frequency_score"] = pd.qcut(dataframe['frequency'].rank(method="first"), 5, [1, 2, 3, 4, 5])
    
    # Calculate the monetary score by dividing 'monetary' into 5 quantiles, assigning scores from 1 to 5 (higher values are better)
    dataframe["monetary_score"] = pd.qcut(dataframe['monetary'], 5, [1, 2, 3, 4, 5])
    
    # Concatenate the 'recency_score' and 'frequency_score' to create the final RFM score
    dataframe["RFM_SCORE"] = (dataframe['recency_score'].astype(str) + dataframe['frequency_score'].astype(str))

    return dataframe



def rfm_score_clustering(dataframe):
    """
    Segments customers into different categories based on their RFM scores using predefined segmentation rules.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing the 'RFM_SCORE' column for each customer.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: The input DataFrame with an additional 'segment' column representing customer segments.
        - dict: A dictionary mapping RFM score patterns to customer segments.
    """
    # Define a mapping of RFM score patterns to customer segments
    seg_map = {
        r'[1-2][1-2]': 'hibernating',  # Low recency and low frequency
        r'[1-2][3-4]': 'at_risk',     # Low recency and moderate frequency
        r'[1-2]5': 'cant_loose',       # Low recency and high frequency
        r'3[1-2]': 'about_to_sleep',   # Moderate recency and low frequency
        r'33': 'need_attention',       # Moderate recency and moderate frequency
        r'[3-4][4-5]': 'loyal_customers', # Moderate to high recency and high frequency
        r'41': 'promising',            # High recency and low frequency
        r'51': 'new_customers',        # High recency and moderate frequency
        r'[4-5][2-3]': 'potential_loyalists', # High recency and low to moderate frequency
        r'5[4-5]': 'champions'         # High recency and high frequency
    }
    
    # Map RFM scores to customer segments using the predefined segmentation rules
    dataframe['segment'] = dataframe['RFM_SCORE'].replace(seg_map, regex=True)
    
    # Reset the DataFrame index
    dataframe.reset_index(inplace=True)

    return dataframe, seg_map



def rfm_segment_analysis(dataframe, palette):
    """
    Generates histograms for the distribution of 'recency', 'monetary', and 'frequency' across different RFM segments.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'recency', 'monetary', 'frequency', and 'segment' columns.
    palette (dict): A dictionary specifying the color palette for the segments.

    Returns:
    None: Displays histograms of the specified features.
    """
    # Create a figure with 3 subplots arranged vertically
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Set the main title for the entire figure
    fig.suptitle('RFM Segment Analysis', size=14)
    
    # List of features to be plotted
    feature_list = ['recency', 'monetary', 'frequency']
    
    # Iterate over the features and their corresponding axes
    for idx, col in enumerate(feature_list):
        sns.histplot(
            ax=axes[idx], 
            data=dataframe,
            hue='segment', 
            x=feature_list[idx],
            palette=palette
        )
        # Set specific x-axis limits for 'monetary' and 'frequency' features
        if idx == 1:
            axes[idx].set_xlim([0, 400])
        if idx == 2:
            axes[idx].set_xlim([0, 30])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Add legend to the top right of the figure
    plt.legend(loc="upper right")
    
    # Display the plots
    plt.show()



def tree_map_segment(dataframe, seg_map):
    """
    Creates a treemap visualization of customer segments based on their counts.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing the 'segment' column with customer segments.
    seg_map (dict): A dictionary mapping RFM score patterns to segment names.

    Returns:
    None: Displays a treemap of customer segments.
    """
    # Calculate the count of customers in each segment and sort in descending order
    segments = dataframe["segment"].value_counts().sort_values(ascending=False)
    
    # Initialize the figure and axis for the plot
    fig = plt.gcf()
    ax = fig.add_subplot()
    
    # Set the figure size
    fig.set_size_inches(16, 10)
    
    # Plot the treemap
    squarify.plot(
        sizes=segments,  # Sizes of the treemap squares based on segment counts
        label=[label for label in seg_map.values()],  # Labels for each segment
        color=[
            "#AFB6B5", "#F0819A", "#926717", "#F0F081",
            "#81D5F0", "#C78BE5", "#748E80", "#FAAF3A",
            "#7B8FE4", "#86E8C0"
        ],  # Colors for the segments
        pad=False,  # Remove padding between segments
        bar_kwargs={"alpha": 1},  # Set transparency of bars to fully opaque
        text_kwargs={"fontsize": 15}  # Set font size for text labels
    )
    
    # Set title and axis labels
    plt.title("Customer Segmentation Map", fontsize=20)
    plt.xlabel("Frequency", fontsize=18)
    plt.ylabel("Recency", fontsize=18)
    
    # Display the plot
    plt.show()



def segment_customers(dataframe):
    """
    Plots a count plot showing the number of customers in each segment and annotates the plot with percentage values.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing a 'segment' column with customer segments.

    Returns:
    None: Displays a count plot of customer segments.
    """
    # Set the figure size
    plt.figure(figsize=(18, 8))
    
    # Define a color palette for the segments
    palette = sns.color_palette("Set1", len(dataframe['segment'].unique()))
    
    # Create a count plot of customer segments
    ax = sns.countplot(
        data=dataframe,
        x='segment',
        palette=palette
    )
    
    # Total number of customers
    total = len(dataframe['segment'])
    
    # Annotate each bar with the percentage of customers
    for patch in ax.patches:
        # Calculate the percentage of customers for each segment
        percentage = '{:.1f}%'.format(100 * patch.get_height() / total)
        # Position the annotation slightly above the bar
        x = patch.get_x() + patch.get_width() / 2 - 0.17
        y = patch.get_y() + patch.get_height() * 1.005
        # Add the annotation to the plot
        ax.annotate(percentage, (x, y), size=14)
    
    # Set plot title and axis labels
    plt.title('Number of Customers by Segments', size=16)
    plt.xlabel('Segment', size=14)
    plt.ylabel('Count', size=14)
    plt.xticks(size=10)
    plt.yticks(size=10)
    
    # Display the plot
    plt.show()



def segment_desc_statistics(dataframe):
    """
    Calculates descriptive statistics for 'recency', 'monetary', and 'frequency' for each customer segment.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'recency', 'monetary', 'frequency', and 'segment' columns.

    Returns:
    pd.DataFrame: A DataFrame with descriptive statistics (mean, standard deviation, max, min) for each segment.
    """
    # Group by 'segment' and calculate descriptive statistics for 'recency', 'monetary', and 'frequency'
    desc_stats = dataframe[['recency', 'monetary', 'frequency', 'segment']]\
        .groupby('segment')\
        .agg({'recency': ['mean', 'std', 'max', 'min'],
              'monetary': ['mean', 'std', 'max', 'min'],
              'frequency': ['mean', 'std', 'max', 'min']})
    
    return desc_stats



def last_purchase(dataframe):
    """
    Computes the number of days since the last purchase for each customer based on their most recent purchase date.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'InvoiceDate' and 'Customer ID' columns.

    Returns:
    pd.DataFrame: A DataFrame with 'Customer ID' and the number of days since their last purchase.
    """
    # Convert 'InvoiceDate' to datetime and extract only the date part
    dataframe['InvoiceDay'] = dataframe['InvoiceDate'].dt.date

    # Find the most recent purchase date for each customer
    customer_data = dataframe.groupby('Customer ID')['InvoiceDay'].max().reset_index()

    # Find the most recent date across the entire dataset
    most_recent_date = dataframe['InvoiceDay'].max()

    # Convert 'InvoiceDay' and 'most_recent_date' to datetime type before subtraction
    customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
    most_recent_date = pd.to_datetime(most_recent_date)

    # Calculate the number of days since the last purchase for each customer
    customer_data['Days_Since_Last_Purchase'] = (most_recent_date - customer_data['InvoiceDay']).dt.days

    # Drop the 'InvoiceDay' column as it is no longer needed
    customer_data.drop(columns=['InvoiceDay'], inplace=True)

    return customer_data



def transaction_and_purchased(dataframe, new_df):
    """
    Adds total transactions and total products purchased for each customer to a new DataFrame.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'Customer ID', 'Invoice', and 'Quantity' columns.
    new_df (pd.DataFrame): A pandas DataFrame with at least a 'Customer ID' column to which new features will be added.

    Returns:
    pd.DataFrame: A DataFrame containing 'Customer ID' along with total transactions and total products purchased.
    """
    # Calculate the total number of unique transactions (invoices) for each customer
    total_transactions = dataframe.groupby('Customer ID')['Invoice'].nunique().reset_index()
    total_transactions.rename(columns={'Invoice': 'Total_Transactions'}, inplace=True)

    # Calculate the total number of products purchased by each customer
    total_products_purchased = dataframe.groupby('Customer ID')['Quantity'].sum().reset_index()
    total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

    # Merge the new features into the new_df DataFrame
    customer_data = pd.merge(new_df, total_transactions, on='Customer ID', how='left')
    customer_data = pd.merge(customer_data, total_products_purchased, on='Customer ID', how='left')

    return customer_data



def total_and_average_spend(dataframe, new_df):
    """
    Calculates total spend and average transaction value for each customer and adds these features to a new DataFrame.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'Customer ID', 'Invoice', and 'total_price' columns.
    new_df (pd.DataFrame): A pandas DataFrame with at least a 'Customer ID' column to which new features will be added.

    Returns:
    pd.DataFrame: A DataFrame containing 'Customer ID', total spend, and average transaction value.
    """
    # Calculate the total number of unique transactions (invoices) for each customer
    total_transactions = dataframe.groupby('Customer ID')['Invoice'].nunique().reset_index()
    total_transactions.rename(columns={'Invoice': 'Total_Transactions'}, inplace=True)

    # Calculate the total spend by each customer
    total_spend = dataframe.groupby('Customer ID')['total_price'].sum().reset_index()
    total_spend.rename(columns={'total_price': 'Total_Spend'}, inplace=True)

    # Calculate the average transaction value for each customer
    average_transaction_value = total_spend.merge(total_transactions, on='Customer ID')
    average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / average_transaction_value['Total_Transactions']

    # Merge the new features into the new_df DataFrame
    customer_data = pd.merge(new_df, total_spend, on='Customer ID', how='left')
    customer_data = pd.merge(customer_data, average_transaction_value[['Customer ID', 'Average_Transaction_Value']], on='Customer ID', how='left')

    return customer_data



def product_diversity(dataframe, new_df):
    """
    Calculates the number of unique products purchased by each customer and adds this feature to a new DataFrame.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'Customer ID' and 'StockCode' columns.
    new_df (pd.DataFrame): A pandas DataFrame with at least a 'Customer ID' column to which the new feature will be added.

    Returns:
    pd.DataFrame: A DataFrame containing 'Customer ID' and the number of unique products purchased.
    """
    # Calculate the number of unique products (StockCode) purchased by each customer
    unique_products_purchased = dataframe.groupby('Customer ID')['StockCode'].nunique().reset_index()
    unique_products_purchased.rename(columns={'StockCode': 'Unique_Products_Purchased'}, inplace=True)

    # Merge the new feature into the new_df DataFrame
    customer_data = pd.merge(new_df, unique_products_purchased, on='Customer ID', how='left')

    return customer_data



def behavior_features(dataframe, new_df):
    """
    Computes behavioral features for each customer including average days between purchases,
    favorite shopping day of the week, and favorite shopping hour of the day.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'Customer ID', 'InvoiceDate', and 'InvoiceDay' columns.
    new_df (pd.DataFrame): A pandas DataFrame with at least a 'Customer ID' column to which the new features will be added.

    Returns:
    pd.DataFrame: A DataFrame containing 'Customer ID' along with behavioral features.
    """
    # Extract the day of the week and hour from 'InvoiceDate'
    dataframe['Day_Of_Week'] = dataframe['InvoiceDate'].dt.dayofweek
    dataframe['Hour'] = dataframe['InvoiceDate'].dt.hour

    # Calculate the average number of days between consecutive purchases for each customer
    days_between_purchases = dataframe.groupby('Customer ID')['InvoiceDay'].apply(lambda x: (x.diff().dropna()).apply(lambda y: y.days))
    average_days_between_purchases = days_between_purchases.groupby('Customer ID').mean().reset_index()
    average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)

    # Find the favorite shopping day of the week for each customer
    favorite_shopping_day = dataframe.groupby(['Customer ID', 'Day_Of_Week']).size().reset_index(name='Count')
    favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('Customer ID')['Count'].idxmax()][['Customer ID', 'Day_Of_Week']]

    # Find the favorite shopping hour of the day for each customer
    favorite_shopping_hour = dataframe.groupby(['Customer ID', 'Hour']).size().reset_index(name='Count')
    favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('Customer ID')['Count'].idxmax()][['Customer ID', 'Hour']]

    # Merge the new features into the new_df DataFrame
    customer_data = pd.merge(new_df, average_days_between_purchases, on='Customer ID', how='left')
    customer_data = pd.merge(customer_data, favorite_shopping_day, on='Customer ID', how='left')
    customer_data = pd.merge(customer_data, favorite_shopping_hour, on='Customer ID', how='left')

    return customer_data



def geographic_features(dataframe, new_df):
    """
    Computes geographic features for each customer including the main country of transactions and whether the customer is from the UK.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'Customer ID' and 'Country' columns.
    new_df (pd.DataFrame): A pandas DataFrame with at least a 'Customer ID' column to which the new features will be added.

    Returns:
    pd.DataFrame: A DataFrame containing 'Customer ID' and a binary indicator for whether the customer is from the UK.
    """
    # Count the number of transactions for each customer in each country
    customer_country = dataframe.groupby(['Customer ID', 'Country']).size().reset_index(name='Number_of_Transactions')

    # Find the country with the maximum number of transactions for each customer
    customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates('Customer ID')

    # Create a binary column indicating whether the customer is from the UK or not
    customer_main_country['Is_UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)

    # Merge the new feature into the new_df DataFrame
    customer_data = pd.merge(new_df, customer_main_country[['Customer ID', 'Is_UK']], on='Customer ID', how='left')

    return customer_data



def seasonality_trends(dataframe, new_df):
    """
    Computes seasonal buying patterns and spending trends for each customer.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing 'Customer ID', 'year', 'month', and 'total_price' columns.
    new_df (pd.DataFrame): A pandas DataFrame with at least a 'Customer ID' column to which the new features will be added.

    Returns:
    pd.DataFrame: A DataFrame containing 'Customer ID', monthly spending statistics, and spending trends.
    """
    # Calculate total monthly spending for each customer
    monthly_spending = dataframe.groupby(['Customer ID', 'year', 'month'])['total_price'].sum().reset_index()

    # Calculate Seasonal Buying Patterns: Mean and Std of monthly spending for each customer
    seasonal_buying_patterns = monthly_spending.groupby('Customer ID')['total_price'].agg(['mean', 'std']).reset_index()
    seasonal_buying_patterns.rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'}, inplace=True)

    # Replace NaN values in Monthly_Spending_Std with 0, implying no variability for customers with only one transaction month
    seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0, inplace=True)

    # Calculate Trends in Spending: Slope of the linear trend line fitted to the customer's spending over time
    def calculate_trend(spend_data):
        # If there are more than one data point, calculate the trend using linear regression
        if len(spend_data) > 1:
            x = np.arange(len(spend_data))
            slope, _, _, _, _ = linregress(x, spend_data)
            return slope
        # If only one data point, no trend can be calculated, hence return 0
        else:
            return 0

    # Apply the calculate_trend function to find the spending trend for each customer
    spending_trends = monthly_spending.groupby('Customer ID')['total_price'].apply(calculate_trend).reset_index()
    spending_trends.rename(columns={'total_price': 'Spending_Trend'}, inplace=True)

    # Merge the new features into the new_df DataFrame
    customer_data = pd.merge(new_df, seasonal_buying_patterns, on='Customer ID', how='left')
    customer_data = pd.merge(customer_data, spending_trends, on='Customer ID', how='left')

    return customer_data



def correlation_analysis(dataframe):
    """
    Plots a heatmap of the correlation matrix for the given DataFrame, excluding the 'Customer ID' column.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing numerical columns to analyze correlations.

    Returns:
    None: Displays a heatmap plot.
    """
    sns.set_style('whitegrid')

    # Calculate the correlation matrix excluding the 'Customer ID' column
    corr = dataframe.drop(columns=['Customer ID']).corr()

    # Define a custom colormap
    colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
    my_cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

    # Create a mask to only show the lower triangle of the matrix
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap=my_cmap, annot=True, center=0, fmt='.2f', linewidths=2)
    plt.title('Correlation Matrix', fontsize=14)
    plt.show()



def scale_method(dataframe):
    """
    Scales numerical features of the DataFrame using StandardScaler, excluding specified columns.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing numerical columns to be scaled.

    Returns:
    pd.DataFrame: A DataFrame with scaled numerical features, excluding specified columns.
    """
    scaler = StandardScaler()

    # List of columns that don't need to be scaled
    columns_to_exclude = ['Customer ID', 'Is_UK', 'Day_Of_Week']

    # List of columns that need to be scaled
    columns_to_scale = dataframe.columns.difference(columns_to_exclude)

    # Copy the DataFrame to avoid modifying the original data
    customer_data_scaled = dataframe.copy()

    # Applying the scaler to the necessary columns in the dataset
    customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])

    return customer_data_scaled



def dimensionality_reduction(dataframe):
    """
    Applies PCA (Principal Component Analysis) to the DataFrame and visualizes the explained variance
    and cumulative explained variance to determine the optimal number of principal components.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame with numerical features to apply PCA on.

    Returns:
    None
    """
    # Set the 'Customer ID' column as the index
    dataframe.set_index('Customer ID', inplace=True)

    # Apply PCA
    pca = PCA().fit(dataframe)

    # Calculate the Cumulative Sum of the Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Set the optimal k value (based on analysis or domain knowledge)
    optimal_k = 6

    # Set seaborn plot style
    sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

    # Plot the cumulative explained variance against the number of components
    plt.figure(figsize=(20, 10))

    # Bar chart for the explained variance of each component
    barplot = sns.barplot(x=list(range(1, len(explained_variance_ratio) + 1)),
                        y=explained_variance_ratio,
                        color='#fcc36d',
                        alpha=0.8)

    # Line plot for the cumulative explained variance
    lineplot, = plt.plot(range(1, len(cumulative_explained_variance) + 1), 
                        cumulative_explained_variance, 
                        marker='o', linestyle='--', 
                        color='#ff6200', linewidth=2)

    # Plot optimal k value line
    optimal_k_line = plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k value = {optimal_k}')

    # Set labels and title
    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Explained Variance', fontsize=14)
    plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

    # Customize ticks and legend
    plt.xticks(range(1, len(cumulative_explained_variance) + 1))
    plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
               labels=['Explained Variance of Each Component', 'Cumulative Explained Variance', f'Optimal k value = {optimal_k}'],
               loc=(0.62, 0.1),
               frameon=True,
               framealpha=1.0,  
               edgecolor='#ff6200')

    # Display the variance values for both graphs on the plots
    x_offset = -0.3
    y_offset = 0.01
    for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
        plt.text(i + 1, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
        if i > 0:
            plt.text(i + 1 + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

    plt.grid(axis='both')
    plt.show()



def highlight_top3_pca(dataframe):
    """
    Applies PCA to the input DataFrame, transforms the data, and highlights the top 3 features for each principal component.
    
    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame with numerical features to apply PCA on.

    Returns:
    pd.DataFrame: The transformed DataFrame with principal components as columns.
    pd.io.formats.style.Styler: A styled DataFrame highlighting the top 3 features for each principal component.
    """
    # Initialize PCA to reduce to 6 components
    pca = PCA(n_components=6)

    # Fit PCA on the original data and transform it to the PCA space
    customer_data_pca = pca.fit_transform(dataframe)

    # Create a DataFrame with the PCA components as columns
    customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

    # Restore the original index from the original DataFrame
    customer_data_pca.index = dataframe.index

    def highlight_top3(column):
        """
        Highlights the top 3 absolute values in the column.
        
        Parameters:
        column (pd.Series): A pandas Series representing a principal component.

        Returns:
        list: A list of styles to apply to the DataFrame.
        """
        top3 = column.abs().nlargest(3).index
        return ['background-color: #ffeacc' if i in top3 else '' for i in column.index]

    # Create DataFrame with PCA component loadings
    pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)],  
                         index=dataframe.columns)

    # Apply highlighting to the PCA component DataFrame
    return customer_data_pca, pc_df.style.apply(highlight_top3, axis=0)



def kmeans_clustering(customer_data_pca):
    """
    Visualizes the optimal number of clusters for KMeans clustering using the Elbow method.
    
    Parameters:
    customer_data_pca (pd.DataFrame): DataFrame containing PCA-transformed customer data.
    
    Returns:
    None: Displays the Elbow plot.
    """
    # Set the style and color palette for the plot
    sns.set(style='darkgrid', rc={'axes.facecolor': '#fcf0dc'})
    sns.set_palette(['#ff6200'])

    # Instantiate the KMeans model
    km = KMeans(init='k-means++', n_init=10, max_iter=100, random_state=0)

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 5))

    # Instantiate the KElbowVisualizer with the KMeans model and a range of k values
    visualizer = KElbowVisualizer(km, k=(2, 15), timings=False, ax=ax)

    # Fit the visualizer with the PCA-transformed customer data
    visualizer.fit(customer_data_pca)

    # Display the Elbow plot
    visualizer.show()



def silhouette_analysis(df, start_k, stop_k, figsize=(15, 16)):
    """
    Perform Silhouette analysis for a range of k values and visualize the results.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data to cluster.
    start_k (int): The starting number of clusters to evaluate.
    stop_k (int): The ending number of clusters to evaluate.
    figsize (tuple): The size of the figure for visualization.
    
    Returns:
    None: Displays the silhouette analysis plots.
    """

    # Set the size of the figure
    plt.figure(figsize=figsize)
    grid = gridspec.GridSpec((stop_k - start_k + 1) // 2 + 1, 2)

    # First plot: Silhouette scores for different k values
    first_plot = plt.subplot(grid[0, :])
    sns.set_palette(['darkorange'])

    silhouette_scores = []

    for k in range(start_k, stop_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=0)
        km.fit(df)
        labels = km.predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)

    best_k = start_k + silhouette_scores.index(max(silhouette_scores))

    first_plot.plot(range(start_k, stop_k + 1), silhouette_scores, marker='o')
    first_plot.set_xticks(range(start_k, stop_k + 1))
    first_plot.set_xlabel('Number of clusters (k)')
    first_plot.set_ylabel('Silhouette score')
    first_plot.set_title('Average Silhouette Score for Different k Values', fontsize=15)

    # Add the optimal k value text to the plot
    optimal_k_text = f'The k value with the highest Silhouette score is: {best_k}'
    first_plot.text(start_k + (stop_k - start_k) / 2, max(silhouette_scores) - 0.05, optimal_k_text,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(facecolor='#fcc36d', edgecolor='#ff6200', boxstyle='round,pad=0.5'))

    # Second plot: Silhouette plots for each k value
    colors = sns.color_palette("bright")

    for i in range(start_k, stop_k + 1):    
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        row_idx, col_idx = divmod(i - start_k, 2)  # Adjust the subplot grid

        ax = plt.subplot(grid[row_idx + 1, col_idx])
        visualizer = SilhouetteVisualizer(km, colors=colors, ax=ax)
        visualizer.fit(df)

        # Add the Silhouette score text to the plot
        score = silhouette_score(df, km.labels_)
        ax.text(0.97, 0.02, f'Silhouette Score: {score:.2f}', fontsize=12,
                ha='right', transform=ax.transAxes, color='red')

        ax.set_title(f'Silhouette Plot for {i} Clusters', fontsize=15)

    plt.tight_layout()
    plt.show()



def clustering_feature(dataframe, customer_data_pca, n_clusters=4, custom_mapping=None):
    """
    Perform KMeans clustering on the PCA-transformed data and update the original and PCA DataFrames with new cluster labels.

    Parameters:
    dataframe (pd.DataFrame): The original DataFrame to which cluster labels will be added.
    customer_data_pca (pd.DataFrame): The PCA-transformed DataFrame to which cluster labels will be added.
    n_clusters (int): The number of clusters for KMeans. Default is 4.
    custom_mapping (dict): A dictionary mapping original labels to new labels. Default is None.

    Returns:
    pd.DataFrame, pd.DataFrame: Updated DataFrames with new cluster labels.
    """
    
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, random_state=0)
    kmeans.fit(customer_data_pca)

    # Get the frequency of each cluster
    cluster_frequencies = Counter(kmeans.labels_)

    # Create a mapping from old labels to new labels based on frequency
    label_mapping = {label: new_label for new_label, (label, _) in enumerate(cluster_frequencies.most_common())}

    # If a custom mapping is provided, apply it
    if custom_mapping:
        label_mapping = {v: k for k, v in custom_mapping.items()}

    # Apply the mapping to get the new labels
    new_labels = np.array([label_mapping[label] for label in kmeans.labels_])

    # Append the new cluster labels back to the original dataset
    dataframe['cluster'] = new_labels

    # Append the new cluster labels to the PCA version of the dataset
    customer_data_pca['cluster'] = new_labels

    return dataframe, customer_data_pca



def clustering_visualization(customer_data_pca, palette='viridis'):
    """
    Visualize the distribution of customers across clusters with a horizontal bar plot.

    Parameters:
    customer_data_pca (pd.DataFrame): DataFrame containing PCA-transformed data with cluster labels.
    palette (str or list): Color palette for the bar plot. Default is 'viridis'.

    Returns:
    None
    """
    
    # Calculate the percentage distribution of each cluster
    cluster_percentage = (customer_data_pca['cluster'].value_counts(normalize=True) * 100).reset_index()
    cluster_percentage.columns = ['Cluster', 'Percentage']
    cluster_percentage.sort_values(by='Cluster', inplace=True)

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 4))
    sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette=palette)

    # Adding percentages on the bars
    for index, value in enumerate(cluster_percentage['Percentage']):
        plt.text(value + 1, index, f'{value:.2f}%', va='center')

    plt.title('Distribution of Customers Across Clusters', fontsize=14)
    plt.xticks(ticks=np.arange(0, 101, 10))
    plt.xlabel('Percentage (%)')
    plt.ylabel('Cluster')

    # Show the plot
    plt.show()



def eval_metric(customer_data_pca):
    """
    Evaluate clustering performance metrics and display them in a table.

    Parameters:
    customer_data_pca (pd.DataFrame): DataFrame containing PCA-transformed data with cluster labels.

    Returns:
    None
    """
    
    num_observations = len(customer_data_pca)

    # Check if 'cluster' column exists and there are clusters
    if 'cluster' not in customer_data_pca.columns or len(customer_data_pca['cluster'].unique()) < 2:
        raise ValueError("Clustering results are not available or there are less than 2 clusters.")

    # Separate the features and the cluster labels
    X = customer_data_pca.drop('cluster', axis=1)
    clusters = customer_data_pca['cluster']

    # Compute the metrics
    try:
        sil_score = silhouette_score(X, clusters)
    except ValueError as e:
        sil_score = 'Error: ' + str(e)

    try:
        calinski_score = calinski_harabasz_score(X, clusters)
    except ValueError as e:
        calinski_score = 'Error: ' + str(e)

    try:
        davies_score = davies_bouldin_score(X, clusters)
    except ValueError as e:
        davies_score = 'Error: ' + str(e)

    # Create a table to display the metrics and the number of observations
    table_data = [
        ["Number of Observations", num_observations],
        ["Silhouette Score", sil_score],
        ["Calinski Harabasz Score", calinski_score],
        ["Davies Bouldin Score", davies_score]
    ]

    # Print the table
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt='pretty'))



def radar_map(dataframe):
    """
    Create radar charts to visualize the centroids of clusters.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing the customer data with clusters.

    Returns:
    None
    """
    
    # Set Customer ID as index
    df_customer = dataframe.set_index('Customer ID')

    # Standardize the data (excluding the 'cluster' column)
    scaler = StandardScaler()
    df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['cluster'], axis=1))

    # Create a new DataFrame with standardized values and add the 'cluster' column back
    df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-1], index=df_customer.index)
    df_customer_standardized['cluster'] = df_customer['cluster']

    # Calculate the centroids of each cluster
    cluster_centroids = df_customer_standardized.groupby('cluster').mean()

    # Function to create a radar chart
    def create_radar_chart(ax, angles, data, color, cluster):
        ax.fill(angles, data, color=color, alpha=0.4)
        ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
        ax.set_title(f'Cluster {cluster}', size=16, color=color, y=1.1)

    # Set data
    labels = np.array(cluster_centroids.columns)
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    labels = np.concatenate((labels, [labels[0]]))
    angles += angles[:1]

    # Initialize the figure
    num_clusters = len(cluster_centroids)
    fig, ax = plt.subplots(figsize=(25, 20), subplot_kw=dict(polar=True), nrows=1, ncols=num_clusters)

    # Define a color palette (extend if necessary)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    # Create radar chart for each cluster
    for i in range(num_clusters):
        data = cluster_centroids.loc[i].tolist()
        data += data[:1]  # Complete the loop
        create_radar_chart(ax[i], angles, data, colors[i], i)

    # Customize the radar charts
    for i in range(num_clusters):
        ax[i].set_xticks(angles[:-1])
        ax[i].set_xticklabels(labels[:-1])
        ax[i].grid(color='grey', linewidth=0.5)

    plt.tight_layout()
    plt.show()



def segmentation_features(dataframe):
    """
    Plots histograms for each feature segmented by clusters.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing the features and cluster labels.

    Returns:
    None
    """
    
    # Define colors for clusters (extend if needed)
    colors = plt.cm.tab10.colors

    # Extract feature columns and unique clusters
    features = dataframe.columns[1:-1]
    clusters = dataframe['cluster'].unique()
    clusters.sort()

    # Create subplots: one row per feature, one column per cluster
    n_rows = len(features)
    n_cols = len(clusters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), squeeze=False)

    # Plot histograms for each feature and cluster
    for i, feature in enumerate(features):
        for j, cluster in enumerate(clusters):
            data = dataframe[dataframe['cluster'] == cluster][feature]
            axes[i, j].hist(data, bins=20, color=colors[j % len(colors)], edgecolor='w', alpha=0.7)
            axes[i, j].set_title(f'Cluster {cluster} - {feature}', fontsize=12)
            axes[i, j].set_xlabel(feature)
            axes[i, j].set_ylabel('Frequency')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
