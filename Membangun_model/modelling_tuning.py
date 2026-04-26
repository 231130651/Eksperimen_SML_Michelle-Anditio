import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

dagshub.init(repo_owner='231130651', repo_name='Eksperimen_SML_Michelle-Anditio', mlflow=True)

# Load data
train_df = pd.read_csv('credit_risk_preprocessing/credit_risk_train.csv')
test_df = pd.read_csv('credit_risk_preprocessing/credit_risk_test.csv')

X_train = train_df.drop(columns=['loan_status'])
y_train = train_df['loan_status']

X_test = test_df.drop(columns=['loan_status'])
y_test = test_df['loan_status']

# Set experiment
mlflow.set_experiment("credit_risk_modelling_tuning")

with mlflow.start_run():
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    
    # Manual logging - params
    mlflow.log_param("n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("random_state", 42)
    mlflow.log_param("cv", 3)
    mlflow.log_param("scoring", "f1")

    # Manual logging - metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Artefak 1 - Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Tuning')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tuning.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix_tuning.png')

    # Artefak 2 - Feature Importance
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance - Tuning')
    plt.tight_layout()
    plt.savefig('feature_importance_tuning.png')
    plt.close()
    mlflow.log_artifact('feature_importance_tuning.png')

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_tuning_model")

    print(f"Best params: {grid_search.best_params_}")
    print("Training selesai! Cek DagsHub Experiments.")