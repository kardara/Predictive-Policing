import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define Kigali locations (Gasabo, Kicukiro, Nyarugenge)
kigali_locations = {
    'Gisozi': (-1.933, 30.073),
    'Remera': (-1.944, 30.103),
    'Kimironko': (-1.925, 30.115),
    'Kacyiru': (-1.932, 30.062),
    'Rusororo': (-1.904, 30.124),
    'Gacuriro': (-1.916, 30.083),
    'Kibagabaga': (-1.923, 30.094),
    'Nduba': (-1.910, 30.070),
    'Jali': (-1.900, 30.090),
    'Kicukiro': (-1.972, 30.112),
    'Gikondo': (-1.964, 30.072),
    'Gatenga': (-1.983, 30.125),
    'Niboye': (-1.990, 30.103),
    'Kigarama': (-1.975, 30.094),
    'Masaka': (-1.962, 30.142),
    'Kanombe': (-1.970, 30.135),
    'Nyamirambo': (-1.985, 30.052),
    'Kimisagara': (-1.974, 30.042),
    'Rwezamenyo': (-1.962, 30.032),
    'Muhima': (-1.953, 30.062),
    'Kigali CBD': (-1.952, 30.058),
    'Nyarugenge': (-1.960, 30.050)
}

crime_types = ['Theft', 'Burglary', 'Assault', 'Robbery', 'Vandalism']
crime_severity = {'Theft': 3, 'Burglary': 5, 'Assault': 7, 'Robbery': 6, 'Vandalism': 2}
genders = ['Male', 'Female', 'Other']
roles = ['Victim', 'Suspect']
weather_conditions = ['Clear', 'Rainy', 'Cloudy', 'Stormy']

# Generate more realistic time distribution
def generate_crime_time():
    hour = random.choices(
        population=[random.randint(0, 5), random.randint(6, 11), random.randint(12, 17), random.randint(18, 23)],
        weights=[0.3, 0.2, 0.2, 0.3],  # Night, morning, afternoon, evening
        k=1
    )[0]
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

# Generate dataset
n_rows = 1000
locations = list(kigali_locations.keys())

data = {
    'crime_id': [f"CR{str(i).zfill(5)}" for i in range(1, n_rows + 1)],
    'location': [random.choice(locations) for _ in range(n_rows)],
    'date': [(datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1825))).strftime('%Y-%m-%d') for _ in range(n_rows)],
    'time': [generate_crime_time() for _ in range(n_rows)],
    'crime_type': [random.choice(crime_types) for _ in range(n_rows)],
    'gender': [random.choice(genders) for _ in range(n_rows)],
    'role': [random.choice(roles) for _ in range(n_rows)],
    'age': [random.randint(15, 80) for _ in range(n_rows)],
    'weather': [random.choices(weather_conditions, weights=[0.5, 0.3, 0.15, 0.05])[0] for _ in range(n_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add severity score based on crime type
df['severity'] = [round(crime_severity[ct] + random.uniform(-1, 1), 2) for ct in df['crime_type']]

# Add latitude and longitude with noise
df['latitude'] = df['location'].map(lambda x: kigali_locations[x][0] + random.uniform(-0.01, 0.01))
df['longitude'] = df['location'].map(lambda x: kigali_locations[x][1] + random.uniform(-0.01, 0.01))

# Save to CSV
df.to_csv('kigali_crime_data.csv', index=False)

# Preview
print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())
