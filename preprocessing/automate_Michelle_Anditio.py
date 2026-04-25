from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import pandas as pd
import numpy as np
import os


def preprocess_data(data, target_column, save_path, file_path):
    data = data.drop_duplicates()
    
    data = data[data['person_age'] < 100]
    data = data[data['person_emp_length'] < 60]

    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)

    column_names = data.columns.drop(target_column)
    df_header = pd.DataFrame(columns=column_names)
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Distribusi setelah SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    dump(preprocessor, save_path)
    print(f"Pipeline disimpan ke: {save_path}")

    output_dir = 'credit_risk_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame(X_train, columns=numeric_features)
    train_df[target_column] = y_train.values
    
    test_df = pd.DataFrame(X_test, columns=numeric_features)
    test_df[target_column] = y_test.values
    
    train_df.to_csv(f'{output_dir}/credit_risk_train.csv', index=False)
    test_df.to_csv(f'{output_dir}/credit_risk_test.csv', index=False)
    
    print(f"Train dan test data disimpan ke: {output_dir}/")

    print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def inference(new_data, load_path, column_names):
    preprocessor = load(load_path)
    print(f"Pipeline preprocessing dimuat dari: {load_path}")
    
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data, columns=column_names)
    
    transformed_data = preprocessor.transform(new_data)
    return transformed_data


data = pd.read_csv('../credit_risk_raw.csv')
X_train, X_test, y_train, y_test = preprocess_data(data, 'loan_status', 'pipeline.joblib', 'columns.csv')

column_names = pd.read_csv('columns.csv').columns.tolist()

X_test_df = pd.DataFrame(X_test, columns=column_names)
transformed_test = inference(X_test_df, 'pipeline.joblib', column_names)
print(f"Data test setelah inference: {transformed_test.shape}")