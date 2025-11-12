import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

print("="*60)
print("DIABETES PREDICTION MODEL TRAINING")
print("="*60)

# 1. Load the dataset
print("\n[1/7] Loading dataset...")
try:
    diabetes_data = pd.read_csv('diabetes_dataset.csv')
    print(f"✓ Dataset loaded successfully: {diabetes_data.shape[0]} rows, {diabetes_data.shape[1]} columns")
except FileNotFoundError:
    print("✗ Error: 'diabetes_dataset.csv' not found!")
    exit()

# 2. Data exploration and preprocessing
print("\n[2/7] Analyzing dataset...")
print(f"Dataset Info:")
print(f"  - Total samples: {len(diabetes_data)}")
print(f"  - Features: {diabetes_data.shape[1] - 1}")
print(f"  - Target distribution:")
print(f"    • Non-Diabetic (0): {sum(diabetes_data['Outcome'] == 0)} ({sum(diabetes_data['Outcome'] == 0)/len(diabetes_data)*100:.1f}%)")
print(f"    • Diabetic (1): {sum(diabetes_data['Outcome'] == 1)} ({sum(diabetes_data['Outcome'] == 1)/len(diabetes_data)*100:.1f}%)")

# Check for missing values
missing_values = diabetes_data.isnull().sum().sum()
print(f"  - Missing values: {missing_values}")

# Check for zero values that might be placeholders (common in diabetes datasets)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"\n  Checking for suspicious zero values:")
for col in zero_cols:
    if col in diabetes_data.columns:
        zero_count = (diabetes_data[col] == 0).sum()
        if zero_count > 0:
            print(f"    • {col}: {zero_count} zeros found")
            # Replace zeros with median for biological impossibility
            if col in ['Glucose', 'BloodPressure', 'BMI']:
                median_val = diabetes_data[diabetes_data[col] != 0][col].median()
                diabetes_data[col] = diabetes_data[col].replace(0, median_val)
                print(f"      → Replaced with median: {median_val:.2f}")

# 3. Separate features (X) and target (Y)
print("\n[3/7] Preparing features and target...")
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']
print(f"✓ Features: {list(X.columns)}")
print(f"✓ Target: Outcome (0=Non-Diabetic, 1=Diabetic)")

# 4. Standardize the data
print("\n[4/7] Standardizing features...")
scaler = StandardScaler()
scaler.fit(X)
X_standardized = scaler.transform(X)

print(f"✓ Standardization complete")
print(f"  - Mean of standardized data: {X_standardized.mean():.6f}")
print(f"  - Std of standardized data: {X_standardized.std():.6f}")

# 5. Split data into training and testing sets
print("\n[5/7] Splitting dataset...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X_standardized, Y, 
    test_size=0.2, 
    stratify=Y, 
    random_state=42
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# 6. Train the Support Vector Classifier with hyperparameter tuning
print("\n[6/7] Training SVM model with hyperparameter tuning...")
print("  This may take a few moments...")

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Perform Grid Search
grid_search = GridSearchCV(
    SVC(random_state=42), 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, Y_train)

# Best model
model = grid_search.best_estimator_
print(f"✓ Best parameters found: {grid_search.best_params_}")

# Cross-validation score
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
print(f"✓ Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. Evaluate the model
print("\n[7/7] Evaluating model performance...")

# Training accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)

# Testing accuracy
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)

print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed classification report
print(f"\n{'='*60}")
print(f"CLASSIFICATION REPORT (Test Set)")
print(f"{'='*60}")
print(classification_report(Y_test, test_predictions, 
                          target_names=['Non-Diabetic (0)', 'Diabetic (1)']))

# Confusion Matrix
print(f"{'='*60}")
print(f"CONFUSION MATRIX (Test Set)")
print(f"{'='*60}")
cm = confusion_matrix(Y_test, test_predictions)
print(f"                Predicted")
print(f"              Non-D  Diabetic")
print(f"Actual Non-D    {cm[0][0]:3d}     {cm[0][1]:3d}")
print(f"       Diabetic {cm[1][0]:3d}     {cm[1][1]:3d}")

# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(Y_test, test_predictions)
recall = recall_score(Y_test, test_predictions)
f1 = f1_score(Y_test, test_predictions)

print(f"\n{'='*60}")
print(f"ADDITIONAL METRICS")
print(f"{'='*60}")
print(f"Precision: {precision:.4f} (Of predicted diabetic, {precision*100:.1f}% were correct)")
print(f"Recall:    {recall:.4f} (Detected {recall*100:.1f}% of actual diabetic cases)")
print(f"F1-Score:  {f1:.4f} (Harmonic mean of precision and recall)")

# 8. Save the trained model and scaler
print(f"\n{'='*60}")
print(f"SAVING MODEL AND SCALER")
print(f"{'='*60}")

# Create directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Save to saved_models directory (for organization)
with open('saved_models/diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("✓ Model saved: saved_models/diabetes_model.pkl")

with open('saved_models/diabetes_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("✓ Scaler saved: saved_models/diabetes_scaler.pkl")

# Also save to root directory (for compatibility with app.py)
with open('diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("✓ Model saved: diabetes_model.pkl (root directory)")

with open('diabetes_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("✓ Scaler saved: diabetes_scaler.pkl (root directory)")

print(f"\n{'='*60}")
print(f"✓ DIABETES MODEL TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"\nModel Summary:")
print(f"  - Algorithm: Support Vector Machine (SVM)")
print(f"  - Kernel: {model.kernel}")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Ready for deployment in app.py")
print(f"{'='*60}\n")