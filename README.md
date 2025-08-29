# 🏠 House Price Prediction – End-to-End ML Pipelines

This repository contains three independent Jupyter Notebooks that build **end-to-end machine learning pipelines** for predicting house prices using the popular Kaggle dataset.  
Each notebook demonstrates different approaches to **EDA, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation**.

---

## 📂 Notebooks

### 1. `house-price-ml-pipeline-ipynb-Alemu.ipynb`
- **Objective**: Build and evaluate multiple regression models for house price prediction.
- **Preprocessing**:
  - Missing value handling (`SimpleImputer`)
  - Standardization (`StandardScaler`)
  - One-hot encoding for categorical features
  - Log transformation of target (`np.log1p`)
- **Models**:
  - Ridge Regression
  - KNN
  - Random Forest
  - Gradient Boosting
  - SVR
  - XGBoost
- **Highlights**:
  - Cross-validation leaderboard (RMSLE, RMSE, R²)
  - Hyperparameter tuning via `RandomizedSearchCV` (RF, GB)
  - Feature importance using permutation importance
- **Outputs**:
  - Final trained model (`joblib.dump`)
  - Submission CSV
  - Interactive plots (Plotly: feature importance, residuals, predictions)

---

### 2. `Basit_Shah_MAchine_learning_project_end_to_end_.ipynb`
- **Objective**: Build a structured, modular ML pipeline with full project workflow.
- **Preprocessing**:
  - EDA (visualization, correlation heatmaps, outlier detection)
  - Feature engineering & cleaning
  - Standard scaling and encoding
  - Target log transformation
- **Models**:
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - (Ensemble focus, tuned with CV)
- **Highlights**:
  - `Pipeline` and `ColumnTransformer` integration
  - KFold cross-validation with RMSLE/ RMSE/ R² metrics
  - Hyperparameter tuning with `RandomizedSearchCV`
  - Clear workflow from raw data → deployment-ready pipeline
- **Outputs**:
  - Model evaluation results
  - Submission CSV

---

### 3. `house_pricing_ml_Bishnu.ipynb`
- **Objective**: Lightweight experimentation with tree-based models.
- **Preprocessing**:
  - Basic cleaning and encoding
  - Limited feature engineering
- **Models**:
  - XGBoost (main focus)
- **Highlights**:
  - 5-fold cross-validation
  - Simpler approach compared to others
- **Outputs**:
  - Submission CSV (for Kaggle upload)

---

## ⚙️ Dependencies

Install required libraries:

\`\`\`bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost plotly joblib
\`\`\`

---

## 🚀 How to Run

1. Clone the repo and navigate to the project folder.  
2. Install dependencies.  
3. Open a notebook in Jupyter/Colab:  
   \`\`\`bash
   jupyter notebook house-price-ml-pipeline-ipynb-Alemu.ipynb
   \`\`\`
4. Run cells sequentially to preprocess data, train models, and evaluate.  
5. The final model and submission CSV will be generated.

---

## 📊 Results Overview
- **Tree-based models (RF, GB, XGBoost)** consistently outperformed linear models.  
- **XGBoost** gave the lowest RMSLE, making it the top performer across notebooks.  
- **Key features** (OverallQual, GrLivArea, GarageCars, etc.) were the strongest predictors.  

---

## 📌 Authors
- **Alemu** – Advanced regression pipeline with strong feature engineering.  
- **Basit Shah** – Modular end-to-end ML project.  
- **Bishnu** – Experimental notebook focusing on XGBoost.  
