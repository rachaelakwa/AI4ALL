import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import csv
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set field size limit to handle large JSON-like strings in the CSV
csv.field_size_limit(sys.maxsize)

# Load a subset of the data
df_train = pd.read_csv("filtered_train.csv", engine='python')
df_test = pd.read_csv("ga-customer-revenue-prediction/test.csv", engine='python')

# Function to clean and extract relevant fields from nested columns
def clean_and_extract_fields(column_name, keys_to_keep):
    # Parse the JSON-like string and retain only the specified keys
    def clean_nested_data(nested_data_str):
        nested_data = json.loads(nested_data_str)
        cleaned_data = {k: nested_data.get(k, None) for k in keys_to_keep}
        return json.dumps(cleaned_data)
    
    # Apply the cleaning function
    df_train[column_name] = df_train[column_name].apply(clean_nested_data)
    
    # Extract the relevant fields into new columns
    def extract_nested_fields(nested_data_str):
        nested_data = json.loads(nested_data_str)
        return pd.Series({k: nested_data.get(k) for k in keys_to_keep})
    
    extracted_df = df_train[column_name].apply(extract_nested_fields)
    return extracted_df

# Define keys to keep for each nested column
totals_keys = ["visits", "hits", "pageviews", "bounces", "newVisits", "transactionRevenue"]
traffic_source_keys = ["campaign", "source", "medium", "keyword", "adContent", "isTrueDirect", "referralPath"]
geo_network_keys = ["continent", "subContinent", "country", "city", "networkDomain"]
device_keys = ["deviceCategory", "isMobile", "operatingSystem", "browser"]


# Clean and extract fields for each nested column
totals_df = clean_and_extract_fields("totals", totals_keys)
print("Columns in totals_df:", totals_df.columns)
# print(totals_df.head())
traffic_source_df = clean_and_extract_fields("trafficSource", traffic_source_keys)
geo_network_df = clean_and_extract_fields("geoNetwork", geo_network_keys)
device_df = clean_and_extract_fields("device", device_keys)

# Concatenate all extracted fields with the original DataFrame
df_train = pd.concat([df_train.drop(columns=["totals", "trafficSource", "geoNetwork", "device"]), 
                      totals_df, traffic_source_df, geo_network_df, device_df], axis=1)

# Identify numerical columns
# Manually classified numerical columns
numerical_cols = [
    'visits', 'hits', 'pageviews', 'bounces', 'newVisits', 'transactionRevenue',
    'visitId', 'visitNumber', 'visitStartTime', 'date'
]

# Manually classified non-numerical columns
non_numerical_cols = [
    'channelGrouping', 'socialEngagementType',
    'campaign', 'source', 'medium', 'keyword', 'adContent',
    'isTrueDirect', 'referralPath', 'continent', 'subContinent',
    'country', 'city', 'networkDomain', 'deviceCategory',
    'isMobile', 'operatingSystem', 'browser'
]


# Identify binary and multi-value non-numerical columns
binary_cols = [col for col in non_numerical_cols if df_train[col].nunique() <= 2]
multi_value_cols = [col for col in non_numerical_cols if df_train[col].nunique() > 2]

# One-hot encode multi-value columns
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_data = one_hot_encoder.fit_transform(df_train[multi_value_cols])
encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(multi_value_cols))
df_train = pd.concat([df_train.drop(columns=multi_value_cols), encoded_df], axis=1)

# Label encode binary columns
label_encoder = LabelEncoder()
for col in binary_cols:
    df_train[col] = label_encoder.fit_transform(df_train[col])

# print(len(df_train.columns))

#checking for null values
# total_nan_count = df_train.isna().sum().sum()
# print("Total number of NaN values in the DataFrame:", total_nan_count)

# total_non_nan_count = df_train.notna().sum().sum()
# print("Total number of non-NaN values in the DataFrame:", total_non_nan_count)

# Replace all NaN values with zero
df_train = df_train.fillna(0)

# print("NaN values have been replaced with zero.")


# Standardizing and dropping irrelevant columns using PCA
features = df_train.drop(columns=['transactionRevenue'], errors='ignore')
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.95)  # Retain 95% of the variance
features_pca = pca.fit_transform(features_scaled)

# Check the number of components chosen to retain 95% of the variance
print("Number of components chosen:", pca.n_components_)

# Create a new DataFrame with the PCA components
df_train_pca = pd.DataFrame(features_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
print(df_train_pca.head())

# Assuming 'target' is the name of your target variable
# Split the data into features and target
print('transactionRevenue' in df_train.columns)
# print("Columns in df_train after concatenation:", df_train.columns)


# y = df_train['transactionRevenue'] 
# X = df_train.drop(columns=['transactionRevenue'], errors='ignore')   

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize the model
# model = RandomForestClassifier(random_state=42)

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Model Accuracy:", accuracy)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
