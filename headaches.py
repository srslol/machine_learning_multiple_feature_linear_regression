import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import datetime

def predict_headache(csv_file, input_date):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file)

    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Create a new column 'true_false' with True/False values
    data['true_false'] = data['event'].astype(bool)

    # Split the data into training and testing sets
    X = data['date']
    y = data['true_false']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform the training and testing data
    X_train_date = [date.toordinal() for date in X_train]
    X_test_date = [date.toordinal() for date in X_test]

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(pd.DataFrame(X_train_date), y_train)

    # Make a prediction for the input date
    input_date = datetime.datetime.strptime(input_date, '%Y-%m-%d')
    input_date_ordinal = input_date.toordinal()

    # Check if the input date is within the trained range
    if input_date_ordinal < min(X_train_date) or input_date_ordinal > max(X_train_date):
        return "Date is outside of trained range"

    prediction = model.predict(pd.DataFrame([input_date_ordinal]))
    return prediction[0]

# Example usage:
csv_file = 'headaches.csv'
input_date = '2024-06-09'
prediction = predict_headache(csv_file, input_date)
print(f"Predicted true/false: {prediction}")