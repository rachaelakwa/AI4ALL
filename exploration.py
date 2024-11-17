import pandas as pd
import json

df_train = pd.read_csv('filtered_train.csv', nrows=30000)
print(df_train.columns)
# for i in df_train.columns:
#     print(df_train[i])


print(df_train['socialEngagementType'])

print(df_train.loc[1, 'totals'])
print('\n')
print(df_train.loc[1, 'trafficSource'])
print('\n')
print(df_train.loc[5000, 'geoNetwork'])
print('\n')
print(df_train.loc[5000, 'device'])
print('\n')



# import json

# # Function to collect all unique keys in a JSON-like column
# def collect_keys(column_name):
#     unique_keys = set()  # Use a set to avoid duplicate keys
#     for entry in df_train[column_name]:
#         try:
#             data = json.loads(entry)
#             unique_keys.update(data.keys())  # Add keys to the set
#         except (json.JSONDecodeError, TypeError):
#             continue  # Skip rows with invalid JSON data
#     return unique_keys

# # Collect keys for each nested column
# totals_keys = collect_keys("totals")
# traffic_source_keys = collect_keys("trafficSource")
# geo_network_keys = collect_keys("geoNetwork")
# device_keys = collect_keys("device")

# # Print all possible keys for each column
# print("Keys in 'totals':", totals_keys)
# print('\n')
# print("Keys in 'trafficSource':", traffic_source_keys)
# print('\n')
# print("Keys in 'geoNetwork':", geo_network_keys)
# print('\n')
# print("Keys in 'device':", device_keys)
# print('\n')

