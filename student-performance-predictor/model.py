import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("dataset.csv")

# Input & output
X = data[['study_hours', 'attendance', 'sleep_hours', 'previous_score']]
y = data['final_score']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")

from sklearn.metrics import r2_score

predictions = model.predict(X_test)
print("Model Accuracy (R2 Score):", r2_score(y_test, predictions))
