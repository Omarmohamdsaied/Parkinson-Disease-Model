# Parkinson's Disease Detection using Machine Learning

This project implements machine learning models for Parkinson's disease detection and analysis through:
- Regression models to predict UPDRS scores
- Classification models for disease diagnosis

## Project Structure

```
.
├── classification/           # Disease classification models
│   ├── dataset/             # Classification datasets
│   ├── model.ipynb          # Classification model notebook
│   └── [docs]              # Project documentation
├── regression/              # UPDRS score prediction
│   ├── dataset/            # Regression datasets
│   ├── parkinsons_disease.py    # Main regression code
│   └── Parkinson's_Disease_notebook.ipynb  # Regression notebook
├── test_script.py          # Testing utilities
└── env.yml                 # Conda environment file
```

## Local Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Parkinson-Disease-Model
   ```

2. **Set up Conda Environment**
   ```bash
   # Create conda environment from file
   conda env create -f env.yml

   # Activate the environment
   conda activate pattern_env
   ```

3. **Project Components**
   - **Regression Analysis**: 
     - Navigate to `regression/` directory
     - Open `Parkinson's_Disease_notebook.ipynb` in Jupyter Notebook/Lab
     - Run `parkinsons_disease.py` for the main regression analysis

   - **Classification Analysis**:
     - Navigate to `classification/` directory
     - Open `model.ipynb` in Jupyter Notebook/Lab

4. **Required Dependencies**
   - numpy
   - pandas
   - matplotlib
   - scikit-learn
   - scikit-image
   - seaborn

## Usage

1. **For Regression Analysis**
   - The regression models predict UPDRS (Unified Parkinson's Disease Rating Scale) scores
   - Use the Jupyter notebook for interactive analysis and visualization
   - Run `parkinsons_disease.py` for the complete regression pipeline

2. **For Classification**
   - The classification models focus on disease diagnosis
   - Follow the `model.ipynb` notebook for step-by-step analysis

## Testing

Run the test suite using:
```bash
python test_script.py
```
