import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
import logging
import sys
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CLASSIFICATION_TARGET = "Diagnosis"

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

def preprocess_classification_data(df, numerical_medians=None):
    """Preprocess the data specifically for classification"""
    try:
        # Store target column if it exists
        target = None
        if CLASSIFICATION_TARGET in df.columns:
            target = df[CLASSIFICATION_TARGET].copy()
            df = df.drop(columns=[CLASSIFICATION_TARGET])
        
        # Convert WeeklyPhysicalActivity to hours
        df['WeeklyPhysicalActivity'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_time_to_hours)
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
        
        # Fill missing numerical values with training medians if provided, else use test data medians
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if numerical_medians is not None:
            for col in numerical_columns:
                if col in numerical_medians:
                    df[col] = df[col].fillna(numerical_medians[col])
                else:
                    df[col] = df[col].fillna(df[col].median())
        else:
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

        # Drop unnecessary columns
        df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

        # Restore target column if it existed
        if target is not None:
            df[CLASSIFICATION_TARGET] = target

        return df
    except Exception as e:
        logging.error(f"Error during data preprocessing: {str(e)}")
        raise

def load_classification_model():
    """Load the saved classification model"""
    try:
        with open('classification/models/classification_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            # Handle both old and new model data formats
            feature_names = model_data.get('expected_columns', model_data.get('feature_names'))
            if feature_names is None:
                raise ValueError("No feature names found in model data")
            numerical_medians = model_data.get('numerical_medians')
            
            logging.info(f"Loaded model with {len(feature_names)} features")
            if numerical_medians is not None:
                logging.info("Loaded numerical medians for preprocessing")
            else:
                logging.warning("No numerical medians found in model data, will use test data medians")
                
        return model, feature_names, numerical_medians
    except Exception as e:
        logging.error(f"Error loading classification model: {str(e)}")
        raise

def test_classification_model(test_data_path):
    """Test classification model on new data"""
    try:
        # Load model and preprocessing information
        model, feature_names, numerical_medians = load_classification_model()
        logging.info("Model loaded successfully")

        # Load and preprocess test data
        test_df = pd.read_csv(test_data_path)
        logging.info("Test data loaded successfully")

        # Store target column if it exists
        target = None
        if CLASSIFICATION_TARGET in test_df.columns:
            target = test_df[CLASSIFICATION_TARGET].copy()

        # Preprocess the data using training medians
        test_df = preprocess_classification_data(test_df, numerical_medians)
        logging.info("Data preprocessing completed")

        # Ensure all expected features are present
        missing_features = set(feature_names) - set(test_df.columns)
        if missing_features:
            logging.warning(f"Adding missing features: {missing_features}")
            for feature in missing_features:
                test_df[feature] = 0

        # Create feature matrix for prediction (excluding target)
        X_test = test_df[feature_names]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # If target column exists, calculate and print metrics
        if target is not None:
            accuracy = accuracy_score(target, y_pred)
            report = classification_report(target, y_pred)
            
            logging.info("Classification Results:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Detailed Classification Report:")
            logging.info(f"\n{report}")
        
        # Save predictions
        pd.DataFrame({'Classification_Predictions': y_pred}).to_csv('classification_predictions.csv', index=False)
        logging.info("Predictions saved to classification_predictions.csv")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_script_cls.py <classification_test_file>")
        sys.exit(1)
    
    test_data_path = sys.argv[1]
    test_classification_model(test_data_path) 