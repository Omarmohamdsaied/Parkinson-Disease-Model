import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
import os
import pickle
import logging

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from sklearn.linear_model import Ridge, Lasso

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
    """Load and preprocess the regression dataset"""
    logging.info("Loading and preprocessing data...")
    try:
        # Load data
        df = pd.read_csv('dataset/parkinsons_disease_data_reg.csv')
        
        # Convert WeeklyPhysicalActivity to hours
        df['WeeklyPhysicalActivity (hr)'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_time_to_hours)
        
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
        X = df.drop(['UPDRS', 'PatientID', 'DoctorInCharge'], axis=1)
        y = df['UPDRS']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, X.columns.tolist(), medians
    
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
    """Train the regression model"""
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names, medians = load_and_preprocess_data()
        
        # Create and train model
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Model Performance:")
        logging.info(f"Mean Squared Error: {mse:.4f}")
        logging.info(f"R2 Score: {r2:.4f}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and preprocessing information
        model_data = {
            'model': reg,
            'feature_names': feature_names,
            'numerical_medians': medians,
            'expected_columns': feature_names,
            'mse': mse,
            'r2': r2
        }
        
        with open('models/regression_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info("Model and preprocessing information saved successfully")
        
        return reg, mse, r2
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

# Reading Data From CSV File
data=pd.read_csv("dataset/parkinsons_disease_data_reg.csv")

"""# **Exploring Data :-->**"""

# Knowing the number of Rows & Columns
data.shape

# Knowing size of Data
data.size

# Exploring first 5 records from Data
data.head()

# To see columns names
data.columns

# To see some information
data.info()

# To see some statistical information
data.drop('PatientID',axis=1).describe()

# To check dataType for each column
data.dtypes

# To check if there is duplication in data or not
data.duplicated().sum()

# To know the total nulls in each column
data.isnull().sum()


data['Gender'].value_counts()

data['Ethnicity'].value_counts()

data['EducationLevel'].value_counts()

data['Smoking'].value_counts()

EducationLevel = data[data['EducationLevel'].isnull()]
EducationLevel_nulls= EducationLevel['EducationLevel'].value_counts().sum()
EducationLevel_nulls

data.isnull().sum()

# calculate % of nulls in data

null_count = data['EducationLevel'].isnull().sum()

total_rows = len(data)

null_percentage = (null_count / total_rows) * 100
print(f"Percentage of nulls in Edu_level column : {null_percentage:.2f}%")

plt.figure(figsize=(20, 8))

# Draw the heatmap
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f")

plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(25,8))
plt.plot(data.corr(numeric_only=True).UPDRS.sort_values(ascending=False)[1:],label="Correlation",c='r',lw=5,marker='o',ms=10)
plt.ylabel("Correlation")
plt.xlabel("Feature")
plt.legend()
plt.grid(True)
plt.show()

# Calculate means
edu_updrs = data.groupby('EducationLevel')['UPDRS'].mean().reset_index()

# Bar plot
plt.figure(figsize=(8,5))
sns.barplot(x='EducationLevel', y='UPDRS', data=edu_updrs, palette='viridis')
plt.title('Average UPDRS by Education Level')
plt.ylabel('Average UPDRS Score')
plt.xlabel('Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,40))
plot=1
columns = data.select_dtypes(include='number').columns
for i in columns:
    if plot<12:
        plt.subplot(9,2,plot)
        sns.kdeplot(data, x=data[i],color='green',shade=1,edgecolor='black')
    plot+=1
plt.show()

"""**Data is normally distributed in the most of the columns**

---
---
"""

sns.countplot(x='Gender', data=data,palette='Set1',edgecolor='black')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

"""**Data seems that it's equally distributed between males and females**

---
---

"""
sns.countplot(x='Ethnicity', data=data,palette='Set2',edgecolor='black')
plt.title('The ethnicity of the patients distribution')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.show()

"""**Most of patients are Caucasian**

---
---
"""

sns.countplot(x='Smoking', data=data,hue='Gender',palette='Set1',edgecolor='black')
plt.title('Smoking distribution')
plt.xlabel('Smoking_status')
plt.ylabel('Count')
plt.show()

"""---
---

"""

data['Age'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

"""**Data is slightly biased to age range from 68 to 72 years**

---
---
"""





"""# **Preprocessing :>**"""

# Converting the string into a real dictionary
data['MedicalHistory'] = data['MedicalHistory'].apply(ast.literal_eval)

# Expand the dictionary into separate columns
medical_columns = data['MedicalHistory'].apply(pd.Series)

# Merge with original DataFrame
data = pd.concat([data, medical_columns], axis=1)

# Optional: drop the original MedicalHistory column
data.drop(columns=['MedicalHistory'], inplace=True)

# Converting the string into a real dictionary
data['Symptoms']=data['Symptoms'].apply(ast.literal_eval)

# Expand the dictionary into separate columns
symptoms_column = data['Symptoms'].apply(pd.Series)

# Merge with original DataFrame
data = pd.concat([data, symptoms_column], axis=1)

# Drop the original MedicalHistory column
data.drop(columns=['Symptoms'], inplace=True)


data.head()



# Checking OUTLIERS

columns = data.select_dtypes(include='number').columns
for col in columns:
    # calculate interquartile range
    q25, q75 = np.percentile(data[col], 25), np.percentile(data[col], 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outliers = ( ( data[col] < lower) | (data[col] > upper))
    index_label = data[outliers].index
    print(f'Number of outliers in {col}: {len(index_label)}')




# Convert HH:MM to number of minutes only
def convert_to_minutes(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

data['WeeklyPhysicalActivity (hr)'] = data['WeeklyPhysicalActivity (hr)'].apply(convert_to_minutes)

data.dtypes

object_columns = data.select_dtypes(include='object').columns.tolist()
object_columns

"""**Dropping columns : :**"""

data.drop(columns=['PatientID','EducationLevel','DoctorInCharge'],inplace=True,axis=1)

"""**Scaling Data : :**"""


# Make a copy
df_encoded = data.copy()

# Separate target
target = df_encoded['UPDRS']

# Identify only numeric feature columns (excluding target)
numeric_features = df_encoded.drop('UPDRS', axis=1).select_dtypes(include=['number']).columns

# Apply log transformation (add small value to avoid log(0) issues)
df_encoded[numeric_features] = np.log1p(df_encoded[numeric_features])

# Create standard scaler
scaler = StandardScaler()

# Apply scaling ONLY on numeric features
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# Put back the target if needed (in case you dropped it earlier)
df_encoded['UPDRS'] = target

# Update original dataframe
data[df_encoded.columns] = df_encoded

# Check the result
pd.set_option('display.max_columns', None)
data.head()

"""**Encoding Data : :**


"""



# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Apply Label Encoding for all categorical columns except 'Ethnicity'
label_encoder = LabelEncoder()

for col in categorical_columns:
    if col != 'Ethnicity':
        data[col] = label_encoder.fit_transform(data[col].astype(str))  # Convert to str to avoid errors with non-string data

# Apply One-Hot Encoding for the 'Ethnicity' column
data['Ethnicity'] = data['Ethnicity'].astype(str)

# Now apply pd.get_dummies() for one-hot encoding
data = pd.get_dummies(data, columns=['Ethnicity'], drop_first=True,dtype=int)

# Now your dataframe has Label Encoded columns, and One-Hot Encoded 'Ethnicity'
data.head()



"""# **Feature Engineer : :**

**They are not usefull logically and at the model in the prediction of UPDRS**
"""

#  Create Binary Count of Symptoms Present (Yes/No)
#data['NumSymptoms'] = data[['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems', 'Constipation']].apply(lambda x: (x == 'Yes').sum(), axis=1)

#  Interaction Features: Age with Symptoms (for example)
#data['AgeTremorInteraction'] = data['Age'] * data['Tremor'].apply(lambda x: 1 if x == 'Yes' else 0)
#data['AgeRigidityInteraction'] = data['Age'] * data['Rigidity'].apply(lambda x: 1 if x == 'Yes' else 0)

#  Interaction Features: BMI with Symptoms
#data['BMI_TremorInteraction'] = data['BMI'] * data['Tremor'].apply(lambda x: 1 if x == 'Yes' else 0)
#data['BMI_RigidityInteraction'] = data['BMI'] * data['Rigidity'].apply(lambda x: 1 if x == 'Yes' else 0)

#  Combine Blood Pressure and Cholesterol (e.g., ratio or other metrics)
#data['BloodPressure'] = data['SystolicBP'] / (data['DiastolicBP'] + 1)  # Added 1 to avoid division by 0
#data['CholesterolBalance'] = data['CholesterolLDL'] / (data['CholesterolHDL'] + 1)

"""# **Feature Selection :>**"""

# Define the feature matrix X and the target y
#data.drop(columns='UPDRS_Category',axis=1,inplace=True)
X = data.drop(columns='UPDRS')
y = data['UPDRS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Create a random forest classifier
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Get feature importances
selecting_features = rf.feature_importances_

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': selecting_features
})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(data=importances_df, x='Importance', y='Feature', color='g',edgecolor='black')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()




# Choose top 10 best features (you can change k)
selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Top Selected Features:")
print(selected_features)

# Create and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Get the feature coefficients
selecting_features = lr.coef_

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(selecting_features)  # Take absolute value
})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(data=importances_df, x='Importance', y='Feature', color='skyblue', edgecolor='black')
plt.title('Feature Importance from Linear Regression')
plt.xlabel('Absolute Coefficient Value (Importance)')
plt.ylabel('Feature')
plt.show()

"""# **Machine Learning Models :>**"""

data_x = data[['WeeklyPhysicalActivity (hr)','CholesterolHDL','BMI','AlcoholConsumption','CholesterolTotal',
              'SleepQuality','FunctionalAssessment','CholesterolLDL','DietQuality' ,'CholesterolTriglycerides','SystolicBP'
              ,'MoCA','DiastolicBP','Age']]

Y = data['UPDRS']

X_train, X_test, y_train, y_test = train_test_split(data_x, Y, test_size = 0.3,shuffle=True,random_state=40)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def save_model(model, model_name, features, metrics_dict):
    """
    Save model and its metadata to a pickle file
    """
    model_data = {
        'model': model,
        'features': features,
        'metrics': metrics_dict,
        'feature_names': list(features.columns) if hasattr(features, 'columns') else None
    }
    
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved {model_name} model to models/{model_name}.pkl")

def linear(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2_test = metrics.r2_score(y_test, y_pred)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    
    print("linear regression:")
    print("MSE: ", mse)
    print("r2_score test:", r2_test)
    print("r2_score train:", r2_train)

    # Save the model
    metrics_dict = {
        'mse': mse,
        'r2_score_test': r2_test,
        'r2_score_train': r2_train
    }
    save_model(model, 'linear_regression', X_train, metrics_dict)

    return mse

LR = linear(X_train, y_train , X_test, y_test)


def poly(degree, X_train, y_train, X_test, y_test):
    poly_features = PolynomialFeatures(degree)
    X_train_poly = poly_features.fit_transform(X_train)

    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    y_pred = poly_model.predict(poly_features.fit_transform(X_test))

    mse = mean_squared_error(y_test, y_pred)
    r2_test = metrics.r2_score(y_test, y_pred)
    r2_train = metrics.r2_score(y_train, y_train_predicted)
    
    print("Polynomial regression:")
    print("MSE: ", mse)
    print("r2_score test:", r2_test)
    print("r2_score train:", r2_train)

    # Save the model
    model_data = {
        'model': poly_model,
        'poly_features': poly_features,
        'metrics': {
            'mse': mse,
            'r2_score_test': r2_test,
            'r2_score_train': r2_train,
            'degree': degree
        }
    }
    with open(f'models/polynomial_regression_deg{degree}.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved polynomial regression model (degree {degree}) to models/polynomial_regression_deg{degree}.pkl")
    
    return mse

poly(3, X_train, y_train , X_test, y_test)

data_x = data[['CholesterolHDL','SleepQuality','MoCA',"Depression",'PosturalInstability']]


Y = data['UPDRS']



X_train, X_test, y_train, y_test = train_test_split(data_x, Y, test_size = 0.3,shuffle=True,random_state=40)


rf_regressor = RandomForestRegressor(max_depth=3 ,max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 5, n_estimators = 200)
rf_regressor.fit(X_train, y_train)


# Predicting on the test set
y_pred = rf_regressor.predict(X_test)
y_train_pred = rf_regressor.predict(X_train)
print("Random forest:")
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("r2_score_test:",metrics.r2_score(y_test,y_pred))
print("r2_score_train:",metrics.r2_score(y_train,y_train_pred))

# Save Random Forest model
metrics_dict = {
    'mse': metrics.mean_squared_error(y_test, y_pred),
    'r2_score_test': metrics.r2_score(y_test, y_pred),
    'r2_score_train': metrics.r2_score(y_train, y_train_pred)
}
save_model(rf_regressor, 'random_forest', X_train, metrics_dict)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # alpha = penalty strength
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)
y_train_pred = ridge_model.predict(X_train)

print("Ridge Regression:")
print("MSE:", round(metrics.mean_squared_error(y_test, y_pred), 4))
print("r2_score_test:", round(metrics.r2_score(y_test, y_pred), 4))
print("r2_score_train:", round(metrics.r2_score(y_train, y_train_pred), 4))

# Save Ridge model
metrics_dict = {
    'mse': metrics.mean_squared_error(y_test, y_pred),
    'r2_score_test': metrics.r2_score(y_test, y_pred),
    'r2_score_train': metrics.r2_score(y_train, y_train_pred),
    'alpha': 1.0
}
save_model(ridge_model, 'ridge_regression', X_train, metrics_dict)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)  # alpha = penalty strength
lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)
y_train_pred = lasso_model.predict(X_train)

print("Lasso Regression:")
print("MSE:", round(metrics.mean_squared_error(y_test, y_pred), 4))
print("r2_score_test:", round(metrics.r2_score(y_test, y_pred), 4))
print("r2_score_train:", round(metrics.r2_score(y_train, y_train_pred), 4))

# Save Lasso model
metrics_dict = {
    'mse': metrics.mean_squared_error(y_test, y_pred),
    'r2_score_test': metrics.r2_score(y_test, y_pred),
    'r2_score_train': metrics.r2_score(y_train, y_train_pred),
    'alpha': 0.1
}
save_model(lasso_model, 'lasso_regression', X_train, metrics_dict)

