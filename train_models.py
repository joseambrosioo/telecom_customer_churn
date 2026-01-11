# train_models.py

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# --- Data Pre-processing ---
def preprocess_data(train_path, test_path):
    telcom = pd.read_csv(train_path)
    telcom_test = pd.read_csv(test_path)
    
    col_to_drop = [
        'State', 'Area code', 'Total day charge', 'Total eve charge',
        'Total night charge', 'Total intl charge'
    ]
    telcom = telcom.drop(columns=col_to_drop, axis=1)
    telcom_test = telcom_test.drop(columns=col_to_drop, axis=1)

    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    bin_cols = [col for col in bin_cols if col != 'Churn']
    
    le = LabelEncoder()
    for col in bin_cols:
        telcom[col] = le.fit_transform(telcom[col])
        telcom_test[col] = le.transform(telcom_test[col])

    num_cols = [col for col in telcom.columns if telcom[col].dtype in ['float64', 'int64'] and col not in bin_cols + ['Churn']]
    std = StandardScaler()
    telcom[num_cols] = std.fit_transform(telcom[num_cols])
    telcom_test[num_cols] = std.transform(telcom_test[num_cols])
    
    target_col = ['Churn']
    cols = [col for col in telcom.columns if col not in target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        telcom[cols], telcom[target_col], test_size=0.25, random_state=111
    )
    
    # Save the scaler and label encoder in the 'data' folder
    joblib.dump(std, 'data/scaler.pkl')
    joblib.dump(le, 'data/label_encoder.pkl')

    return telcom, telcom_test, X_train, X_test, y_train, y_test, cols

if __name__ == '__main__':
    # Create 'models' and 'data' folders if they do not exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('data'):
        os.makedirs('data')

    telcom, telcom_test, X_train, X_test, y_train, y_test, cols = preprocess_data(
        'dataset/churn-bigml-80.csv', 'dataset/churn-bigml-20.csv'
    )

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=9, random_state=42),
        'KNN Classifier': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'SVM (RBF)': SVC(C=10.0, gamma=0.1, probability=True, random_state=42),
        'LGBM Classifier': LGBMClassifier(learning_rate=0.5, max_depth=7, n_estimators=100, random_state=42, verbosity=-1),
        'XGBoost Classifier': XGBClassifier(learning_rate=0.9, max_depth=7, n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42),
        'Bagging Classifier': BaggingClassifier(random_state=42),
    }

    # Train and save all models
    model_results = {}
    print("Starting training and saving models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train.values.ravel())
        joblib.dump(model, f'models/{name.replace(" ", "_")}.pkl')
        model_results[name] = model
    print("Training and saving completed.")
    
    # Save the list of columns for later use
    joblib.dump(cols, 'data/features.pkl')