import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import hdbscan


def parse_windows_security_events(xml_path):
    ns = {'ev': 'http://schemas.microsoft.com/win/2004/08/events/event'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for event in root.findall('ev:Event', ns):
        rec = {}
        system = event.find('ev:System', ns)
        if system is not None:
            rec['EventID'] = system.findtext('ev:EventID', default='', namespaces=ns)
            time_elem = system.find('ev:TimeCreated', ns)
            rec['TimeCreated'] = time_elem.get('SystemTime') if time_elem is not None else None
            rec['EventRecordID'] = system.findtext('ev:EventRecordID', default='', namespaces=ns)
            provider = system.find('ev:Provider', ns)
            rec['ProviderName'] = provider.get('Name') if provider is not None else None

        eventdata = event.find('ev:EventData', ns)
        if eventdata is not None:
            for data in eventdata.findall('ev:Data', ns):
                name = data.get('Name')
                rec[name] = data.text

        records.append(rec)

    return pd.DataFrame(records)


xml_path = r"C:\Users\ASUS\Desktop\AIC211-20260512T030119Z-3-001\AIC211\log5hour.xml"
df = parse_windows_security_events(xml_path)

# Add derived time features for behavior clustering
if 'TimeCreated' in df.columns:
    df['TimeCreated'] = pd.to_datetime(df['TimeCreated'], errors='coerce')
    df['HourOfDay'] = df['TimeCreated'].dt.hour
    df['DayOfWeek'] = df['TimeCreated'].dt.dayofweek

behavior_columns = [
    'EventID',
    'ProviderName',
    'SubjectUserName',
    'SubjectDomainName',
    'TargetUserName',
    'TargetDomainName',
    'AccountName',
    'IpAddress',
    'WorkstationName',
    'LogonType',
    'HourOfDay',
    'DayOfWeek',
]

behavior_features = [col for col in behavior_columns if col in df.columns]

if len(behavior_features) == 0:
    raise ValueError('No behavior features found in the parsed events. Check the XML field names.')

# Separate numeric and categorical features
numeric_features = [col for col in behavior_features if df[col].dtype.kind in 'biufc']
categorical_features = [col for col in behavior_features if col not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='drop',
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
])

X = pipeline.fit_transform(df[behavior_features].fillna(''))

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_method='leaf',
)

labels = clusterer.fit_predict(X)
df['BehaviorCluster'] = labels

print('Behavior features used:', behavior_features)
print('Cluster counts:')
print(pd.Series(labels).value_counts().sort_index())
print(df[['EventID', 'BehaviorCluster']].head(20))

# Notes:
# - HDBSCAN labels of -1 mean the event did not fit into a stable cluster.
# - This script clusters events by behavior, not by anomaly detection.
# - You can adjust `min_cluster_size` and `min_samples` to change cluster granularity.
