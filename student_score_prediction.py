import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


print("Student Score Prediction")

# Load dataset
data = pd.read_csv("student_score.csv")
print(data.head())

# Features and target
X = data[["StudyHours", "Attendance", "PreviousScore"]]
y = data["FinalScore"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Actual Scores:", y_test.values)
print("Predicted Scores:", y_pred)

# Example prediction
new_student = pd.DataFrame(
    [[6, 90, 75]],
    columns=["StudyHours", "Attendance", "PreviousScore"]
)

predicted_score = model.predict(new_student)

print("Example Predicted Final Score:", predicted_score[0])

# Save trained model
joblib.dump(model, "student_score_model.pkl")

# User input
print("\nEnter information for a new student:")

study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous_score = float(input("Enter Previous Score: "))

new_student = pd.DataFrame(
    [[study_hours, attendance, previous_score]],
    columns=["StudyHours", "Attendance", "PreviousScore"]
)

new_prediction = model.predict(new_student)

print("Predicted Final Score:", new_prediction[0])