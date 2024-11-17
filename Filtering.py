import pandas as pd
import json

# Load the original dataset
df_original = pd.read_csv("ga-customer-revenue-prediction/train.csv", engine='python')

# Function to check if 'transactionRevenue' exists and is not zero in the 'totals' column
def has_transaction_revenue(totals_str):
    try:
        totals = json.loads(totals_str)
        # Check if 'transactionRevenue' is present and greater than zero
        return 'transactionRevenue' in totals and totals['transactionRevenue'] != "0"
    except (json.JSONDecodeError, TypeError):
        return False

# Apply the filter
df_filtered = df_original[df_original['totals'].apply(has_transaction_revenue)]

# Export the filtered DataFrame to a new CSV file
df_filtered.to_csv("filtered_train.csv", index=False)

print("Filtered data exported successfully to 'filtered_train.csv'.")
