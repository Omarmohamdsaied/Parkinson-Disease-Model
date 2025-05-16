import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import logging
import sys
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

REGRESSION_TARGET = "UPDRS"
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

def preprocess_data(df, is_training=False):
    """Preprocess the data with proper handling of missing values"""
    try:
        # Convert WeeklyPhysicalActivity to hours
        df['WeeklyPhysicalActivity'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_time_to_hours)
        df = df.drop(columns=['WeeklyPhysicalActivity (hr)'])
        
        # Handle categorical columns
        categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking']
        for col in categorical_columns:
            if col not in df.columns:
                df[col] = 'Unknown'
            df[col] = df[col].fillna('Unknown')
        
        # Expand dictionary columns
        df = expand_dict_column(df, 'MedicalHistory')
        df = expand_dict_column(df, 'Symptoms')
        
        # Fill missing numerical values with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        logging.error(f"Error during data preprocessing: {str(e)}")
        raise

def load_models():
    """Load the saved regression and classification models with their preprocessing objects"""
    try:
        # Load regression model
        with open('regression/models/regression_model.pkl', 'rb') as f:
            regression_data = pickle.load(f)
            regression_model = regression_data['model']
            regression_preprocessor = regression_data.get('preprocessor')
            regression_features = regression_data.get('feature_names')
        
        # Load classification model
        with open('classification/models/classification_model.pkl', 'rb') as f:
            classification_data = pickle.load(f)
            classification_model = classification_data['model']
            classification_preprocessor = classification_data.get('preprocessor')
            classification_features = classification_data.get('feature_names')
        
        return (regression_model, regression_preprocessor, regression_features,
                classification_model, classification_preprocessor, classification_features)
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def test_models(test_regression_path, test_classification_path):
    """Test both regression and classification models on new data"""
    try:
        # Load models and preprocessors
        (regression_model, regression_preprocessor, regression_features,
         classification_model, classification_preprocessor, classification_features) = load_models()
        
        # Test regression model
        regression_df = pd.read_csv(test_regression_path)
        regression_df = preprocess_data(regression_df)
        
        if regression_features is not None:
            regression_df = regression_df[regression_features]
        
        if regression_preprocessor is not None:
            X_regression = regression_preprocessor.transform(regression_df)
        else:
            X_regression = regression_df
            
        y_regression_pred = regression_model.predict(X_regression)
        
        if REGRESSION_TARGET in regression_df.columns:
            y_regression_true = regression_df[REGRESSION_TARGET]
            mse = mean_squared_error(y_regression_true, y_regression_pred)
            r2 = r2_score(y_regression_true, y_regression_pred)
            logging.info(f"Regression Results:")
            logging.info(f"Mean Squared Error: {mse:.4f}")
            logging.info(f"R2 Score: {r2:.4f}")
        
        # Test classification model
        classification_df = pd.read_csv(test_classification_path)
        classification_df = preprocess_data(classification_df)
        
        if classification_features is not None:
            classification_df = classification_df[classification_features]
        
        if classification_preprocessor is not None:
            X_classification = classification_preprocessor.transform(classification_df)
        else:
            X_classification = classification_df
            
        y_classification_pred = classification_model.predict(X_classification)
        
        if CLASSIFICATION_TARGET in classification_df.columns:
            y_classification_true = classification_df[CLASSIFICATION_TARGET]
            accuracy = accuracy_score(y_classification_true, y_classification_pred)
            report = classification_report(y_classification_true, y_classification_pred)
            logging.info(f"Classification Results:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Classification Report:\n{report}")
        
        # Save predictions
        pd.DataFrame({'Regression_Predictions': y_regression_pred}).to_csv('regression_predictions.csv', index=False)
        pd.DataFrame({'Classification_Predictions': y_classification_pred}).to_csv('classification_predictions.csv', index=False)
        
        logging.info("Predictions saved to regression_predictions.csv and classification_predictions.csv")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_script.py <regression_test_file> <classification_test_file>")
        sys.exit(1)
    
    test_regression_path = sys.argv[1]
    test_classification_path = sys.argv[2]
    test_models(test_regression_path, test_classification_path)