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
        # Get all unique keys
        all_keys = set()
        for d in df[column].dropna():
            all_keys.update(d.keys())
        
        # Create binary columns
        for key in all_keys:
            col_name = f"{column}_{key}"
            df[col_name] = df[column].apply(lambda x: 1 if isinstance(x, dict) and x.get(key) == 'Yes' else 0)
        
        # Drop original column
        df = df.drop(columns=[column])
    except Exception as e:
        logging.warning(f"Error expanding column {column}: {str(e)}")
    return df

def load_and_preprocess_data():
    """Load and preprocess the classification dataset"""
    logging.info("Loading and preprocessing data...")
    try:
        # Load data
        df = pd.read_csv('../data/classification_data.csv')
        
        # Convert WeeklyPhysicalActivity to hours
        df['WeeklyPhysicalActivity'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_time_to_hours)
        df = df.drop(columns=['WeeklyPhysicalActivity (hr)'])
        
        # Handle categorical columns
        categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking']
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Expand dictionary columns
        df = expand_dict_column(df, 'MedicalHistory')
        df = expand_dict_column(df, 'Symptoms')
        
        # Fill missing numerical values with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Split features and target
        X = df.drop(['Diagnosis', 'PatientID', 'DoctorInCharge'], axis=1)
        y = df['Diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    except Exception as e:
        logging.error(f"Error during data loading and preprocessing: {str(e)}")
        raise

def create_preprocessor(X_train):
    """Create a preprocessor for the features"""
    try:
        # Identify numeric and categorical columns
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns
        
        # Create preprocessing steps for both numeric and categorical features
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    except Exception as e:
        logging.error(f"Error creating preprocessor: {str(e)}")
        raise

def train_model():
    """Train the classification model"""
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
        
        # Create and fit preprocessor
        preprocessor = create_preprocessor(X_train)
        
        # Create pipeline
        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        logging.info("Training model...")
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
            'accuracy': accuracy
        }
        
        with open('models/classification_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info("Model saved successfully")
        
        return clf, accuracy
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()