import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature

dagshub.init(repo_owner='231130651', repo_name='Eksperimen_SML_Michelle-Anditio', mlflow=True)

# Load data
train_df = pd.read_csv('credit_risk_preprocessing/credit_risk_train.csv')
test_df = pd.read_csv('credit_risk_preprocessing/credit_risk_test.csv')

X_train = train_df.drop(columns=['loan_status'])
y_train = train_df['loan_status']

X_test = test_df.drop(columns=['loan_status'])
y_test = test_df['loan_status']

# Set experiment
mlflow.set_experiment("credit_risk_modelling")

# Autolog
mlflow.autolog()

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Artefak 1 - Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')

    # Artefak 2 - Feature Importance
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    mlflow.log_artifact('feature_importance.png')

    print("Training selesai! Cek DagsHub Experiments.")