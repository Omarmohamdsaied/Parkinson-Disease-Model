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
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_time_to_hours(time_str):
    """Convert time string in format HH:MM to hours as float"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes/60
    except:
        return np.nan

def expand_dict_column(df, column):
    """Expand a dictionary column into multiple binary columns"""
    try:
        # Convert string representation of dict to actual dict
        df[column] = df[column].apply(ast.literal_eval)
        
        # Define expected keys for each dictionary column
        expected_keys = {
            'MedicalHistory': ['FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 
                             'Diabetes', 'Depression', 'Stroke'],
            'Symptoms': ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability',
                        'SpeechProblems', 'SleepDisorders', 'Constipation']
        }
        
        # Create binary columns for expected keys
        if column in expected_keys:
            for key in expected_keys[column]:
                col_name = key
                df[col_name] = df[column].apply(lambda x: 1 if isinstance(x, dict) and x.get(key) == 'Yes' else 0)
        
        # Drop original column
        df = df.drop(columns=[column])
    except Exception as e:
        logging.warning(f"Error expanding column {column}: {str(e)}")
    return df

def create_categorical_features(df):
    """Create standardized categorical features"""
    # Create specific dummy variables for Gender
    df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
    
    # Create specific dummy variables for Ethnicity
    ethnicity_mapping = {
        'Asian': 'Ethnicity_Asian',
        'Caucasian': 'Ethnicity_Caucasian'
    }
    for eth, col_name in ethnicity_mapping.items():
        df[col_name] = (df['Ethnicity'] == eth).astype(int)
    df['Ethnicity_Other'] = (~df['Ethnicity'].isin(['Asian', 'Caucasian'])).astype(int)
    
    # Create specific dummy variables for EducationLevel
    education_mapping = {
        'High School': 'EducationLevel_High School',
        'Higher': 'EducationLevel_Higher'
    }
    for edu, col_name in education_mapping.items():
        df[col_name] = (df['EducationLevel'] == edu).astype(int)
    df['EducationLevel_nan'] = (df['EducationLevel'] == 'Unknown').astype(int)
    
    # Create specific dummy variable for Smoking
    df['Smoking_Yes'] = (df['Smoking'] == 'Yes').astype(int)
    
    # Drop original categorical columns
    df = df.drop(columns=['Gender', 'Ethnicity', 'EducationLevel', 'Smoking'])
    
    return df

def load_and_preprocess_data():
    """Load and preprocess the classification dataset"""
    logging.info("Loading and preprocessing data...")
    try:
        # Load data
        df = pd.read_csv('dataset/parkinsons_disease_data_cls.csv')
        
        # Convert WeeklyPhysicalActivity to hours
        df['WeeklyPhysicalActivity (hr)'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_time_to_hours)
        df = df.drop(columns=['WeeklyPhysicalActivity (hr)'])
        
        # Handle categorical columns
        categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking']
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Create standardized categorical features
        df = create_categorical_features(df)
        
        # Expand dictionary columns
        df = expand_dict_column(df, 'MedicalHistory')
        df = expand_dict_column(df, 'Symptoms')
        
        # Fill missing numerical values with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        medians = df[numerical_columns].median()
        df[numerical_columns] = df[numerical_columns].fillna(medians)
        
        # Split features and target
        X = df.drop(['Diagnosis', 'PatientID', 'DoctorInCharge'], axis=1)
        y = df['Diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, X.columns.tolist(), medians
    
    except Exception as e:
        logging.error(f"Error during data loading and preprocessing: {str(e)}")
        raise

def train_model():
    """Train the classification model"""
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names, medians = load_and_preprocess_data()
        
        # Create and train model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logging.info(f"Model Accuracy: {accuracy:.4f}")
        logging.info(f"Classification Report:\n{report}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and preprocessing information
        model_data = {
            'model': clf,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'numerical_medians': medians,
            'expected_columns': feature_names
        }
        
        with open('models/classification_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info("Model and preprocessing information saved successfully")
        
        return clf, accuracy
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()