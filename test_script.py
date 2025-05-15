import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

def load_models():
    """Load the saved regression and classification models"""
    try:
        with open('regression/regression_model.pkl', 'rb') as f:
            regression_model = pickle.load(f)
        with open('classification/classification_model.pkl', 'rb') as f:
            classification_model = pickle.load(f)
        return regression_model, classification_model
    except FileNotFoundError:
        print("Error: Model files not found. Please ensure models are trained and saved.")
        return None, None

def preprocess_test_data(data):
    """Handle missing values and prepare test data"""
    # Fill missing values with median for numerical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Fill missing categorical values with mode
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    return data

def evaluate_models(reg_model, clf_model, X_reg_test, y_reg_test, X_clf_test, y_clf_test):
    """Evaluate models and print metrics"""
    # Regression metrics
    reg_predictions = reg_model.predict(X_reg_test)
    mse = mean_squared_error(y_reg_test, reg_predictions)
    r2 = r2_score(y_reg_test, reg_predictions)
    
    # Classification metrics
    clf_predictions = clf_model.predict(X_clf_test)
    accuracy = accuracy_score(y_clf_test, clf_predictions)
    
    print("\nModel Performance Metrics:")
    print("========================")
    print("Regression Model:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("\nClassification Model:")
    print(f"Accuracy: {accuracy:.4f}")

def main():
    # Load the saved models
    regression_model, classification_model = load_models()
    if regression_model is None or classification_model is None:
        return
    
    try:
        # Load test datasets
        regression_test = pd.read_csv('regression_test.csv')
        classification_test = pd.read_csv('classification_test.csv')
        
        # Preprocess test data
        X_reg_test = preprocess_test_data(regression_test.drop('target', axis=1))
        y_reg_test = regression_test['target']
        
        X_clf_test = preprocess_test_data(classification_test.drop('target', axis=1))
        y_clf_test = classification_test['target']
        
        # Evaluate models
        evaluate_models(
            regression_model, 
            classification_model,
            X_reg_test, 
            y_reg_test,
            X_clf_test, 
            y_clf_test
        )
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()