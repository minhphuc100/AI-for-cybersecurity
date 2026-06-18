import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

df = pd.read_csv('networklog.csv')
print(df['Source'].value_counts())  # Xem phân bố lớp

# them count column cho moi Source
df['count'] = df.groupby('Source')['Source'].transform('count')
print(df[['Source','Length','count']].drop_duplicates())
# convert char to numeric for Isolation Forest
df['Source'] = df['Source'].astype('category').cat.codes
df['Protocol'] = df['Protocol'].astype('category').cat.codes

print(df[['Source','Protocol','Length','count']].drop_duplicates())



# convert df to numpy array for Isolation Forest
X = df[['Source','Protocol','count','Length']]  # Use 'count', 'Length' as the features for anomaly detection

# Initialize and fit the Isolation Forest model
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict anomalies
anomaly_labels = iso_forest.predict(X)

# Add anomaly labels to the DataFrame
df['anomaly'] = anomaly_labels
# 1 for normal, -1 for anomaly
# Print the results
print(df[['Source', 'count', 'Length', 'anomaly']].drop_duplicates())
