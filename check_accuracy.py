import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt  # type: ignore
import random

# === Rule-based prediction function (tuned) ===
def rule_based_prediction(row, add_noise=False):
    score = 0
    if row["BMI"] > 30: score += 2
    if row["Age"] > 50: score += 1
    if row["Blood Pressure Level"] == "High": score += 1
    if row["Weight Change"] == "Yes": score += 1
    if row["Smoking"] == "Yes": score += 1
    if row["Family History"] in ["Mother", "Father", "Both Parents"]: score += 2
    if row["Frequent Urination"] == "Yes": score += 2
    if row["Increased Thirst"] == "Yes": score += 1
    if row["Diseases"] == "Yes": score += 1

    prediction = "High" if score >= 5 else "Low"

    # Optional noise to prevent overfitting (1% chance)
    if add_noise and random.random() < 0.01:
        prediction = "Low" if prediction == "High" else "High"

    return prediction

# === Load dataset ===
df = pd.read_csv("Final_Diabetes_Prediction_Dataset_with_Videos.csv")

if "Diabetes Risk" not in df.columns:
    print("❌ 'Diabetes Risk' column not found in dataset.")
    exit()

# === Split into training and test sets ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# === Predict on test set ===
test_df["Predicted Risk"] = test_df.apply(lambda row: rule_based_prediction(row, add_noise=True), axis=1)

# === Evaluation metrics ===
y_true = test_df["Diabetes Risk"]
y_pred = test_df["Predicted Risk"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="High")
recall = recall_score(y_true, y_pred, pos_label="High")
f1 = f1_score(y_true, y_pred, pos_label="High")
cm = confusion_matrix(y_true, y_pred, labels=["High", "Low"])

print(f"\n📊 Evaluation on Test Set:")
print(f"✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"🎯 Precision: {precision * 100:.2f}%")
print(f"🔁 Recall:    {recall * 100:.2f}%")
print(f"📐 F1 Score:  {f1 * 100:.2f}%")

print("\n🧮 Confusion Matrix (Actual vs Predicted):")
print(cm)

# === Optional confusion matrix plot ===
show_plot = input("\nWould you like to see the confusion matrix plot? (y/n): ").strip().lower()
if show_plot == "y":
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["High", "Low"])
    disp.plot()
    plt.show()
