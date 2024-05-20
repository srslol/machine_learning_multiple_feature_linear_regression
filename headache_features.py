import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Read the data
df = pd.read_csv('headaches.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Ensure data is sorted by date
df = df.sort_values('date')

# Check the initial data
print("Initial DataFrame:\n", df.head())
print("Total rows in initial DataFrame:", len(df))

# Step 2: Create features
# We will use a rolling window to create features for the model
df['headaches_last_7_days'] = df['event'].rolling(window=7).sum().shift(1).fillna(0)

# Ensure there are no NaN values for other features
df[['sugar', 'sleep', 'sun']] = df[['sugar', 'sleep', 'sun']].fillna(method='ffill')

# Drop rows with NaN values that result from shifting
df = df.dropna()

# Check the data after feature creation
print("DataFrame after feature creation:\n", df.head(10))  # Show more rows for better debugging
print("Total rows after feature creation:", len(df))

# Check the length of the DataFrame
if len(df) < 1:
    raise ValueError("Not enough data to create features. Ensure there are at least 8 rows in the CSV file.")

# Step 3: Train the model
# Define the features and target
features = ['headaches_last_7_days', 'sugar', 'sleep', 'sun']
X = df[features]
y = df['event']

# Check if there are enough samples
if len(X) < 1:
    raise ValueError("Not enough data to create training and testing sets.")

# Perform cross-validation to check model performance
model = LogisticRegression()
cross_val_scores = cross_val_score(model, X, y, cv=5)
print('Cross-validation Accuracy:', cross_val_scores.mean())

# Train the logistic regression model
model.fit(X, y)

# Step 4: Make Predictions
def predict_headache(date, sugar, sleep, sun, model, df):
    # Convert the input date to datetime
    date = pd.to_datetime(date)
    
    # Check if the date is in the dataframe range
    if date not in df['date'].values:
        print("Date is not in the training data range. Predictions might be less accurate.")
    
    # Get the headaches in the last 7 days
    last_7_days = df[df['date'] < date].tail(7)
    
    # If there are not enough data points, return a default value
    if len(last_7_days) < 7:
        print("Not enough historical data to make a prediction for the given date.")
        return False
    
    # Create the feature for the input date
    headaches_last_7_days = last_7_days['event'].sum()
    input_features = pd.DataFrame([[headaches_last_7_days, sugar, sleep, sun]], columns=features)
    
    # Predict the result
    prediction = model.predict(input_features)
    
    return bool(prediction[0])

# Example usage
date_to_predict = '2025-11-29'
sugar_level = 5.5
sleep_hours = 7
sun_exposure = 3
prediction = predict_headache(date_to_predict, sugar_level, sleep_hours, sun_exposure, model, df)
print(f'Will I have a headache on {date_to_predict}? {"Yes" if prediction else "No"}')
