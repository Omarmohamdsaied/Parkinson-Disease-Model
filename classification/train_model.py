import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import ast

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    # Read the dataset
    df = pd.read_csv('dataset/parkinsons_disease_data_cls.csv')
    
    # Convert time format (HH:MM) to hours as float
    def time_to_hours(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes/60
    
    # Convert WeeklyPhysicalActivity to hours
    df['WeeklyPhysicalActivity'] = df['WeeklyPhysicalActivity (hr)'].apply(time_to_hours)
    
    # Handle dictionary columns
    def parse_dict_column(value):
        # Convert string representation of dict to actual dict
        dict_data = ast.literal_eval(value)
        # Convert Yes/No to 1/0
        return {k: 1 if v == 'Yes' else 0 for k, v in dict_data.items()}
    
    # Parse MedicalHistory and Symptoms columns
    medical_history_df = pd.DataFrame([parse_dict_column(x) for x in df['MedicalHistory']])
    symptoms_df = pd.DataFrame([parse_dict_column(x) for x in df['Symptoms']])
    
    # Prepare features
    # First, get the basic numeric columns
    numeric_columns = ['Age', 'BMI', 'AlcoholConsumption', 'DietQuality', 'SleepQuality',
                      'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                      'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
                      'FunctionalAssessment', 'WeeklyPhysicalActivity']
    
    X_numeric = df[numeric_columns]
    
    # Handle categorical columns
    categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking']
    X_categorical = pd.get_dummies(df[categorical_columns], drop_first=True)
    
    # Combine all features
    X = pd.concat([X_numeric, X_categorical, medical_history_df, symptoms_df], axis=1)
    
    # Target variable
    y = df['Diagnosis']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to dataframe to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    print("Training Random Forest model...")
    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model performance...")
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png')
    plt.close()

def save_model(model, scaler):
    print("Saving model and scaler...")
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel and scaler saved successfully!")

def main():
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Save model and scaler
    save_model(model, scaler)

if __name__ == "__main__":
    main()