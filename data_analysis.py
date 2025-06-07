import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import folium
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('kigali_crime_data.csv')
# Preview
print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# --- Preprocessing ---
# Encode categorical variables
le_location = LabelEncoder()
le_crime = LabelEncoder()
le_gender = LabelEncoder()
le_role = LabelEncoder()

df['location_encoded'] = le_location.fit_transform(df['location'])
df['crime_type_encoded'] = le_crime.fit_transform(df['crime_type'])
df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['role_encoded'] = le_role.fit_transform(df['role'])

# Extract month and hour
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['time'] = pd.to_datetime(df['time'], format='%H:%M')
df['hour'] = df['time'].dt.hour

# Create age groups (keep original age for analysis)
df['age_group'] = pd.cut(df['age'], bins=[14, 30, 60, 80], labels=['Younger (15-30)', 'Adult (31-60)', 'Elderly (61-80)'])

# Normalize features for modeling only
scaler = StandardScaler()
X_scaled = df[['age', 'severity']].copy()
X_scaled = scaler.fit_transform(X_scaled)
df['age_scaled'] = X_scaled[:, 0]
df['severity_scaled'] = X_scaled[:, 1]

# --- Classification Model: Predict Crime Type ---
# Features: location, month, hour, gender, role, age, severity
X = df[['location_encoded', 'month', 'hour', 'gender_encoded', 'role_encoded', 'age_scaled', 'severity_scaled']]
y = df['crime_type_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# Logistic Regression
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)

# Evaluate models
print("Random Forest Accuracy (Crime Type):", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred, target_names=le_crime.classes_))
print("Logistic Regression Accuracy (Crime Type):", accuracy_score(y_test, lr_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_pred, target_names=le_crime.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Location', 'Month', 'Hour', 'Gender', 'Role', 'Age', 'Severity'],
    'Importance': rf_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Crime Type):\n", feature_importance)

# --- Classification Model: Predict High Severity ---
# Create binary severity label (high: >5, low: <=5)
df['high_severity'] = (df['severity'] > 5).astype(int)
y_severity = df['high_severity']

# Split data
X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X, y_severity, test_size=0.2, random_state=42)

# Random Forest for severity
rf_sev_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sev_clf.fit(X_train_sev, y_train_sev)
rf_sev_pred = rf_sev_clf.predict(X_test_sev)

# Evaluate severity model
print("\nRandom Forest Accuracy (High Severity):", accuracy_score(y_test_sev, rf_sev_pred))
print("Random Forest Classification Report (High Severity):\n", classification_report(y_test_sev, rf_sev_pred, target_names=['Low', 'High']))

# --- Pattern Analysis ---
# Crimes by location and month
crime_by_location_month = df.groupby(['location', 'month', 'crime_type']).size().unstack(fill_value=0)
print("\nCrimes by Location and Month:\n", crime_by_location_month)

# Crimes by gender and location
crime_by_gender_location = df.groupby(['location', 'gender']).size().unstack(fill_value=0)
print("\nCrimes by Gender and Location:\n", crime_by_gender_location)

# Crimes by age group and location
crime_by_age_location = df.groupby(['location', 'age_group']).size().unstack(fill_value=0)
print("\nCrimes by Age Group and Location:\n", crime_by_age_location)

# Crimes by role and location
crime_by_role_location = df.groupby(['location', 'role']).size().unstack(fill_value=0)
print("\nCrimes by Role and Location:\n", crime_by_role_location)

# --- Visualizations (Simple and Clear) ---
# 1. Crime Types by Location
plt.figure(figsize=(12, 6))
sns.countplot(x='location', hue='crime_type', data=df, palette='Set2')
plt.title('Crime Types by Location', fontsize=16, weight='bold')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Crime Type', fontsize=10)
plt.tight_layout()
plt.savefig('crime_by_location.png')
plt.close()

# 2. Crime Types by Month
plt.figure(figsize=(10, 6))
sns.countplot(x='month', hue='crime_type', data=df, palette='Set2')
plt.title('Crime Types by Month', fontsize=16, weight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.legend(title='Crime Type', fontsize=10)
plt.tight_layout()
plt.savefig('crime_by_month.png')
plt.close()

# 3. Crime Hotspots Map
m = folium.Map(location=[-1.95, 30.05], zoom_start=12)  # Center on Kigali
colors = ['red', 'blue', 'green', 'purple', 'orange']
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=colors[row['cluster']],
        fill=True,
        fill_color=colors[row['cluster']],
        popup=f"Location: {row['location']}, Crime: {row['crime_type']}, Severity: {row['severity']}"
    ).add_to(m)
m.save('kigali_crime_hotspots.html')

# 4. Crimes by Gender and Location
plt.figure(figsize=(12, 6))
crime_by_gender_location.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Crimes by Gender and Location', fontsize=16, weight='bold')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Gender', fontsize=10)
plt.tight_layout()
plt.savefig('crime_by_gender_location.png')
plt.close()

# 5. Crimes by Age Group and Location
plt.figure(figsize=(12, 6))
crime_by_age_location.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Crimes by Age Group and Location', fontsize=16, weight='bold')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Age Group', fontsize=10)
plt.tight_layout()
plt.savefig('crime_by_age_location.png')
plt.close()