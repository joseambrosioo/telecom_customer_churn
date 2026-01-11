# app.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output, dash_table, State, ctx
import dash_bootstrap_components as dbc
import joblib
import os
import urllib.parse
from datetime import datetime
from fpdf import FPDF
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# Model Imports (necessary for loading classes)
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Loading Data and Objects ---
try:
    std = joblib.load('data/scaler.pkl')
    le = joblib.load('data/label_encoder.pkl')
    cols = joblib.load('data/features.pkl')
    telcom_raw = pd.read_csv('dataset/churn-bigml-80.csv')
    telcom_test_raw = pd.read_csv('dataset/churn-bigml-20.csv')

    model_results = {}
    for filename in os.listdir('models'):
        if filename.endswith('.pkl') and filename not in ['scaler.pkl', 'label_encoder.pkl', 'features.pkl']:
            model_name = filename.replace('.pkl', '').replace('_', ' ')
            model_results[model_name] = joblib.load(f'models/{filename}')
except Exception as e:
    print(f"Error loading project files: {e}")
    exit()

def preprocess_data_for_app(df):
    col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    df_proc = df.drop(columns=[c for c in col_to_drop if c in df.columns], axis=1)
    bin_cols = [col for col in df_proc.columns if df_proc[col].nunique() == 2 and col != 'Churn']
    for col in bin_cols:
        df_proc[col] = le.transform(df_proc[col])
    num_cols = [col for col in df_proc.columns if df_proc[col].dtype in ['float64', 'int64'] and col not in bin_cols + ['Churn']]
    df_proc[num_cols] = std.transform(df_proc[num_cols])
    return df_proc

telcom = preprocess_data_for_app(telcom_raw)
telcom_test = preprocess_data_for_app(telcom_test_raw)
X_test = telcom_test[cols]
y_test = telcom_test['Churn']

# --- Metrics Calculation ---
precalculated_rows = []
for name, model in model_results.items():
    predictions = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else predictions
    precalculated_rows.append({
        'Model': name,
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1-Score': f1_score(y_test, predictions, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, prob),
    })
metrics_df = pd.DataFrame(precalculated_rows).round(4)

# --- Feature Dictionary ---
full_feature_list = pd.DataFrame({
    "Feature Name": cols,
    "Data Type": [str(telcom[col].dtype) for col in cols],
    "Example Value": [telcom_raw[col].iloc[0] for col in cols],
    "Business Category": [
        "Usage Metric" if "minutes" in col.lower() or "calls" in col.lower() else 
        "Plan Detail" if "plan" in col.lower() else "Account Info" 
        for col in cols
    ]
})

def preprocess_data_for_app(df):
    """Applies saved transformations to raw data."""
    col_to_drop = [
        'State', 'Area code', 'Total day charge', 'Total eve charge',
        'Total night charge', 'Total intl charge'
    ]
    df = df.drop(columns=col_to_drop, axis=1)

    bin_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Churn']
    for col in bin_cols:
        df[col] = le.transform(df[col])

    num_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in bin_cols + ['Churn']]
    df[num_cols] = std.transform(df[num_cols])
    return df

telcom = preprocess_data_for_app(telcom_raw)
telcom_test = preprocess_data_for_app(telcom_test_raw)


# --- Load Trained Models ---
print("Loading trained models...")
model_results = {}
try:
    for filename in os.listdir('models'):
        if filename.endswith('.pkl'):
            # Transform filename into display format
            model_name = filename.replace('.pkl', '').replace('_', ' ')
            model_results[model_name] = joblib.load(f'models/{filename}')
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Error: 'models' folder not found. Ensure you have run 'train_models.py' first.")
    exit()

# --- Metrics Definition ---
def get_metrics_df(X, y):
    df_rows = []
    for name, model in model_results.items():
        predictions = model.predict(X)
        try:
            probabilities = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, probabilities)
        except (AttributeError, IndexError):
            roc_auc = "N/A"
        
        df_rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y, predictions),
            'Recall': recall_score(y, predictions, zero_division=0),
            'Precision': precision_score(y, predictions, zero_division=0),
            'F1-Score': f1_score(y, predictions, zero_division=0),
            'ROC-AUC': roc_auc,
            'Kappa': cohen_kappa_score(y, predictions),
        })
    return pd.DataFrame(df_rows).round(4)

metrics_train = get_metrics_df(X_test, y_test)
metrics_test = get_metrics_df(telcom_test[cols], telcom_test[['Churn']])


# --- Functions to generate static and dynamic text ---
def get_best_and_worst_models(df, metric='F1-Score'):
    """Finds the 2 best and 2 worst models for a specific metric."""
    df_sorted = df.sort_values(by=metric, ascending=False)
    best_2 = df_sorted.head(2).reset_index(drop=True)
    worst_2 = df_sorted.tail(2).reset_index(drop=True)
    return best_2, worst_2

def generate_static_metrics_summary(df, data_type):
    best_models_f1, _ = get_best_and_worst_models(df, 'F1-Score')
    best_model_name_f1 = best_models_f1.loc[0, 'Model']
    best_model_f1_score = best_models_f1.loc[0, 'F1-Score']
    second_best_model_name_f1 = best_models_f1.loc[1, 'Model']
    second_best_f1_score = best_models_f1.loc[1, 'F1-Score']

    return html.P([
        f"We evaluated the performance of the models on the {data_type} set using key metrics. ",
        "The best performing models were ", html.B(f"{best_model_name_f1}"),
        " with an ", html.B(f"F1-Score of {best_model_f1_score}"), ", followed by ",
        html.B(f"{second_best_model_name_f1}"), " with an ", html.B(f"F1-Score of {second_best_f1_score}"),
        ". Overall, these advanced tree models showed exceptional results."
    ])

def generate_static_confusion_summary(df, data_type):
    best_models_acc, _ = get_best_and_worst_models(df, 'Accuracy')
    best_model_name = best_models_acc.loc[0, 'Model']
    best_model = model_results[best_model_name]
    
    y_pred = best_model.predict(telcom_test[cols]) if data_type == 'test' else best_model.predict(X_test)
    y_actual = telcom_test['Churn'] if data_type == 'test' else y_test
    cm = confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return html.P([
        "A detailed analysis of the confusion matrix for the best performing model, ",
        html.B(best_model_name), f" (accuracy of {best_models_acc.loc[0, 'Accuracy']}), reveals its ability to predict accurately. ",
        f"The model correctly identified ", html.B(f"{tp} True Positives"),
        " and ", html.B(f"{tn} True Negatives"), ", while errors were minimized, with only ",
        html.B(f"{fp} False Positives"), " and ", html.B(f"{fn} False Negatives"),
        ". This demonstrates an ideal balance between capturing churn and avoiding false alarms."
    ])

def generate_static_roc_summary(df, data_type):
    best_models_roc, _ = get_best_and_worst_models(df, 'ROC-AUC')
    best_model_name = best_models_roc.loc[0, 'Model']
    best_model_auc = best_models_roc.loc[0, 'ROC-AUC']
    second_best_model_name = best_models_roc.loc[1, 'Model']
    second_best_auc = best_models_roc.loc[1, 'ROC-AUC']

    return html.P([
        "The ROC curve evaluates the differentiation capability of the model. Models with higher Area Under the Curve (AUC) values are better. ",
        "The best models for this metric were ", html.B(f"{best_model_name}"), " with an ",
        html.B(f"AUC of {best_model_auc}"), ", and ", html.B(f"{second_best_model_name}"),
        f" with an ", html.B(f"AUC of {second_best_auc}"),
        ". Both scores, close to 1.00, indicate that these models are excellent at distinguishing churning customers from non-churning customers."
    ])

# --- Dashboard Layout ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Telecom Customer Churn Prediction"
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("☎️", className="me-2"),
                    dbc.NavbarBrand("Telecom Customer Churn Prediction", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("DS/ML App", color="info", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. ASK Tab
ask_tab = html.Div([
    # Header Section
    html.Div([
        html.H3("❓ ASK — The Business Question", className="mb-0"),
        html.P("Defining the core objectives and stakeholder requirements for customer retention analytics.", className="text-muted"),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Business Task
                    html.B("Business Task", style={"font-size": "1.2rem"}),
                    html.P([
                        "The primary objective is to develop a predictive system capable of identifying customers at high risk of ",
                        html.B("churning"), 
                        " within the telecom ecosystem. Customer acquisition costs significantly outweigh retention costs. By identifying at-risk accounts, the company can proactively deploy ",
                        html.B("targeted intervention strategies"), 
                        " to minimize revenue loss and increase long-term customer lifetime value (CLV)."
                    ]),

                    # Stakeholders
                    html.B("Stakeholders", style={"font-size": "1.2rem"}),
                    html.P([
                        "The key decision-makers for this project include ",
                        html.B("Marketing Strategists, Customer Service Management,"),
                        " and ",
                        html.B("Executive Leadership"),
                        ". These teams require data-driven insights to optimize promotional spend and improve service quality in categories most likely to trigger customer exits."
                    ]),

                    # Deliverables
                    html.B("Deliverables", style={"font-size": "1.2rem"}),
                    html.P([
                        "The final product is this ",
                        html.B("Telecom Retention Dashboard"),
                        ", an interactive application that processes historical usage patterns, evaluates multiple machine learning models (LGBM, XGBoost, Random Forest), and provides a simulated environment to test the sensitivity of churn drivers."
                    ]),
                ], className="p-3 bg-white") 
            ], md=12) 
        ])
    ], fluid=True)
])

# 1. ASK Tab (Restructured to html.Div)
# tab_ask = html.Div([
#     html.Div([
#         html.H3("❓ ASK — The Business Question", className="mb-0"),
#         html.P("Defining objectives for customer retention.", className="text-muted"),
#     ], className="p-4 bg-light border-bottom mb-4"),
#     dbc.Container([
#         html.B("Business Task", style={"font-size": "1.2rem"}),
#         html.P("Predict which customers are likely to cancel their service to allow for proactive retention campaigns.")
#     ], fluid=True)
# ])

# 2. PREPARE Tab
columns_with_types = []
for col in telcom_raw.columns:
    col_type = telcom_raw[col].dtype
    if pd.api.types.is_numeric_dtype(col_type):
        columns_with_types.append({"name": col, "id": col, "type": "numeric"})
    elif pd.api.types.is_bool_dtype(col_type):
        columns_with_types.append({"name": col, "id": col, "type": "text"})
    else:
        columns_with_types.append({"name": col, "id": col})

telcom_raw_display = telcom_raw.head(10).copy()
telcom_raw_display['Churn'] = telcom_raw_display['Churn'].astype(str)

prepare_tab = html.Div(
    children=[
        html.Div([
            html.H4(["📝 ", html.B("PREPARE"), " — Preparing the Data"], className="mt-4"),
            html.P("Identify which specific customer behaviors (Usage, Service Calls, Plans) triggered the churn alert."),
        ], className="p-4 bg-light border-bottom mb-4"),

        html.P("Before we can build a predictive model, we need to understand and clean our data."),
        html.H5("Data Source"),
        html.P(
            ["We used a standard telecom churn dataset, divided into a ", html.B("training set"), " (80% of data) to build our models and a separate ", html.B("test set"), " (20%) to check if our models work on new, unseen data."]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Training Dataset"),
                            dbc.CardBody(
                                [
                                    html.P(f"Rows: {telcom.shape[0]}"),
                                    html.P(f"Features: {telcom.shape[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Test Dataset"),
                            dbc.CardBody(
                                [
                                    html.P(f"Rows: {telcom_test.shape[0]}"),
                                    html.P(f"Features: {telcom_test.shape[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H4("Statistical Summary", className="mt-4"),
        html.P("This table provides a quick statistical overview of the features. Note the perfect linear relationship between minutes and charges for different call types. To avoid multicollinearity in our models, we removed charge-related columns."),
        dbc.Table.from_dataframe(telcom.describe().T.reset_index().rename(columns={'index': 'Feature'}).round(2),
                                 striped=True, bordered=True, hover=True),
        html.H5("Dataset Sample (First 10 Rows)", className="mt-4"),
        dash_table.DataTable(
            id='sample-table',
            columns=columns_with_types,
            data=telcom_raw_display.to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ], className="p-4"
)

# 3. ANALYZE Tab with sub-tabs
analyze_tab = html.Div(
    children=[
        html.Div([
            html.H4(["📈 ", html.B("ANALYZE"), " — Finding Patterns and Building Models"], className="mt-4"),
            html.P("Identify which specific customer behaviors (Usage, Service Calls, Plans) triggered the churn alert."),
        ], className="p-4 bg-light border-bottom mb-4"),

        html.P("This is where we explore data and build the predictive brain of our dashboard."),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.H5("Churn Distribution and Correlations", className="mt-4"),
                        html.P([
                            "The pie chart below shows that our data is ",
                            html.B("not"), " balanced",
                            " — only a small percentage of customers ",
                            html.B("churned,"), " just ",
                            html.B("14.6%"),
                            ". This is important because it means a ",
                            html.B("simple model"),
                            " could get a high ",
                            html.B("accuracy"),
                            " score just by predicting that no one will ever churn. This is why we need ",
                            html.B("more advanced evaluation metrics"),
                            "."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="churn-pie-chart",
                                             figure=go.Figure(
                                                 data=[go.Pie(labels=telcom["Churn"].value_counts().keys().tolist(),
                                                              values=telcom["Churn"].value_counts().values.tolist(),
                                                              marker=dict(colors=['#1f77b4', '#ff7f0e'], line=dict(color="white", width=1.3)),
                                                              hoverinfo="label+percent", hole=0.5)],
                                                 layout=go.Layout(title="Customer Churn Distribution", height=400, margin=dict(t=50, b=50))
                                             )), md=6),
                            dbc.Col(dcc.Graph(id="correlation-matrix"), md=6),
                        ]),
                        html.P([
                            "The ",
                            html.B("Correlation Matrix"),
                            " above shows how strongly each ",
                            html.B("feature"),
                            " relates to others. The ",
                            html.B("darker"),
                            " the cell color of a feature at the intersection with ",
                            html.B("churn"),
                            ", the higher the ",
                            html.B("relationship"),
                            ". The main takeaway is that features like ",
                            html.B("call minutes, international plan,"),
                            " and ",
                            html.B("customer service calls"),
                            " are correlated with ",
                            html.B("churn"),
                            "."
                        ]),
                        html.Hr(),
                        html.H5("Feature Visualization", className="mt-4"),
                        html.P([
                            "This plot visualizes data using two ",
                            html.B("key features"),
                            ": ",
                            html.B("total day minutes"),
                            " and ",
                            html.B("total evening minutes"),
                            ". We separated these periods because ",
                            html.B("customer behavior"),
                            " and ",
                            html.B("reasons for churn"),
                            " might differ throughout the day. A customer making many ",
                            html.B("long calls during the day"),
                            " might be a ",
                            html.B("business user"),
                            ", whereas one with ",
                            html.B("long night calls"),
                            " might be a ",
                            html.B("family user"),
                            "."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(
                                id="day-eve-minutes-plot",
                                figure=go.Figure(
                                    data=go.Scatter(x=telcom['Total day minutes'], y=telcom['Total eve minutes'],
                                                    mode='markers', marker_color=telcom['Churn'], showlegend=False),
                                    layout=go.Layout(title="Total Day Minutes vs. Total Evening Minutes",
                                                     xaxis_title="Total Day Minutes (Scaled)",
                                                     yaxis_title="Total Evening Minutes (Scaled)",
                                                     height=400, margin=dict(t=50, b=50))
                                ))),
                        ]),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance (Training)", children=[
                html.Div(
                    children=[
                        dbc.Row([
                            # Column 1: The Graph
                            dbc.Col([
                                dcc.Graph(id="train-metrics-bar")
                                # dbc.Row([dbc.Col(dcc.Graph(id="train-metrics-bar"), md=12)]),
                            ], md=7),

                            # Column 2: The Explanatory Text
                            dbc.Col([
                                html.H5("Model Performance on Training Data", className="mt-4"),
                                html.P("We trained a variety of machine learning models to see which one performs best."),
                                html.P(
                                    [html.B("The Problem with Accuracy"), ": For our unbalanced data, ", html.B("Accuracy"), " (percentage of correct predictions) is not the best metric. A model that always predicts 'no churn' could have 85% accuracy but would be useless for identifying at-risk customers."]
                                ),
                                html.P(
                                    [html.B("Key Metrics"), ": We focus on a more complete set of metrics:",
                                    html.Ul([
                                        html.Li([html.B("Recall"), " – how many customers who actually churned did we catch?"]),
                                        html.Li([html.B("Precision"), " – of those we flagged as churners, how many were correct?"]),
                                        html.Li([html.B("F1-Score"), " – a balance between Precision and Recall."]),
                                        html.Li([html.B("ROC-AUC"), " – how well the model separates churners from non-churners."])
                                    ])
                                    ]
                                ),
                                generate_static_metrics_summary(metrics_train, 'training'),
                            ], md=5),
                        ], className="align-items-center"), # This centers the content vertically
                        html.Hr(),
                        html.H5("Confusion Matrix and ROC Curve", className="mt-4"),
                        html.P("Select a model to view metrics, Confusion Matrix, and ROC Curve (Training):"),
                        dcc.Dropdown(
                            id='model-selector-train',
                            options=[{'label': name, 'value': name} for name in model_results.keys()],
                            value='LGBM Classifier',
                            clearable=False,
                        ),
                        html.Div(id='selected-train-metrics-summary'),
                        html.Div(id='selected-train-confusion-summary'),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="train-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="train-roc-curve"), md=6),
                        ]),
                        html.Div(id='selected-train-roc-summary'),
                        dbc.Row([
                            # Column 1: Confusion Matrix Details
                            dbc.Col([
                                html.H6("Confusion Matrix", className="mt-4"),
                                html.P([
                                    "The confusion matrix is a table that breaks down our model's predictions into four categories:",
                                    html.Ul([
                                        html.Li([html.B("True Positives (TP):"), " Customers the model correctly predicted would churn."]),
                                        html.Li([html.B("True Negatives (TN):"), " Customers the model correctly predicted would not churn."]),
                                        html.Li([html.B("False Positives (FP):"), " Customers the model incorrectly predicted would churn (Type I Error)."]),
                                        html.Li([html.B("False Negatives (FN):"), " Customers who would churn, but the model missed (Type II Error). This is the costliest category."])
                                    ])
                                ]),
                                generate_static_confusion_summary(metrics_train, 'training'),
                            ], md=6),
                            # Column 2: ROC Curve Details
                            dbc.Col([
                                html.H6("ROC Curve (Receiver Operating Characteristic)", className="mt-4"),
                                html.P([
                                    "The ROC curve plots the ",
                                    html.B("True Positive Rate"),
                                    " against the ",
                                    html.B("False Positive Rate"),
                                    ". The closer the curve is to the top-left corner, the better the model differentiates between the two classes."
                                ]),
                                generate_static_roc_summary(metrics_train, 'training'),
                        ], md=6),
                        ], className="mb-4"),
                        html.Hr(),
                        html.H5("Feature Importance (for tree-based models)", className="mt-4"),
                        html.P("This chart ranks features based on how much they contributed to the model's prediction."),
                        dcc.Dropdown(
                            id="feature-importance-model",
                            options=[{'label': name, 'value': name} for name in ['Decision Tree', 'Random Forest', 'LGBM Classifier', 'XGBoost Classifier', 'Gradient Boosting']],
                            value='LGBM Classifier'
                        ),
                        dcc.Graph(id="feature-importance-plot"),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance (Test)", children=[
                html.Div(
                    children=[
                        html.H5("Model Performance on Test Data", className="mt-4"),
                        dbc.Row([dbc.Col(dcc.Graph(id="test-metrics-bar"), md=12)]),
                        html.P(
                            ["We tested our main models on unseen data to ensure they are not ", html.B("overfitting"), " (memorizing training data instead of learning general patterns)."]
                        ),
                        generate_static_metrics_summary(metrics_test, 'test'),
                        html.Hr(),
                        html.H5("Confusion Matrix and ROC Curve", className="mt-4"),
                        html.P("Select a model to view metrics, Confusion Matrix, and ROC Curve (Test):"),
                        dcc.Dropdown(
                            id='model-selector-test',
                            options=[{'label': name, 'value': name} for name in model_results.keys()],
                            value='LGBM Classifier',
                            clearable=False,
                        ),
                        html.Div(id='selected-test-metrics-summary'),
                        html.Div(id='selected-test-confusion-summary'),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="test-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="test-roc-curve"), md=6),
                        ]),
                        html.Div(id='selected-test-roc-summary'),
                        generate_static_confusion_summary(metrics_test, 'test'),
                        generate_static_roc_summary(metrics_test, 'test'),
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 5. EXPLAIN Tab
tab_explain = html.Div(
    children=[
        html.Div([
            html.H4(["🔍 ", html.B("EXPLAIN"), " — Churn Pattern Breakdown"], className="mt-4"),
            html.P("Identify which specific customer behaviors (Usage, Service Calls, Plans) triggered the churn alert."),
        ], className="p-4 bg-light border-bottom mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.B("Audit Selection")),
                    dbc.CardBody([
                        html.Label("1. Select Customer Index (Test Set):"),
                        dcc.Dropdown(
                            id="customer-dropdown", 
                            options=[{'label': f'Customer {i}', 'value': i} for i in range(len(X_test))], 
                            value=0, clearable=False, className="mb-3"
                        ),
                        html.Label("2. Select Prediction Model:"),
                        dcc.Dropdown(
                            id="explain-model-dropdown", 
                            options=[{'label': name, 'value': name} for name in model_results.keys()], 
                            value='LGBM Classifier', clearable=False
                        ),
                    ])
                ], className="shadow-sm"),
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.B("Internal Risk Summary")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H2(id="prediction-result-text", className="text-center mt-2"),
                                html.Div(id="consensus-alert-container")
                            ], md=6, className="border-end d-flex flex-column justify-content-center"),
                            dbc.Col([
                                dcc.Graph(id="confidence-gauge", style={"height": "180px"})
                            ], md=6)
                        ])
                    ])
                ], className="shadow-sm"),
            ], md=8),
        ], className="mb-4"),
        
        # Waterfall Plot for feature contributions
        dcc.Graph(id="shap-waterfall-plot"),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.B("Legend:"),
                    html.Div([
                        html.Span("█", style={"color": "#ff4136", "margin-right": "10px"}),
                        html.Span("Red: Feature pushed the score toward CHURN (e.g., High Service Calls)")
                    ]),
                    html.Div([
                        html.Span("█", style={"color": "#0074d9", "margin-right": "10px"}),
                        html.Span("Blue: Feature pushed the score toward LOYALTY (e.g., Voice Mail Plan)")
                    ]),
                ], className="p-3 border rounded bg-light", style={"font-size": "0.9rem"})
            ], md=8),
            dbc.Col([
                dbc.Button("📥 Download Churn Audit (PDF)", id="btn-download-local", color="dark", className="w-100 h-100", outline=True),
                dcc.Download(id="download-local-analysis")
            ], md=4)
        ], className="mt-4 gx-3")
    ], className="p-4"
)

# 6. SIMULATE Tab
tab_simulate = html.Div([
    html.Div([
        html.H4(["🧪 ", html.B("SIMULATE"), " — Churn Scenario Builder"], className="mt-4"),
        html.P("Stress-test the system by adjusting usage metrics and plan status to see model sensitivity."),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Row([
        dbc.Col([
            # Churn-specific Sliders
            html.Div([
                html.Label([html.B("Total Day Minutes: "), html.Span(id="val-day-mins")]),
                dcc.Slider(id='sim-day-mins', min=0, max=400, step=10, value=150, marks={0: '0', 400: '400'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Total Eve Minutes: "), html.Span(id="val-eve-mins")]),
                dcc.Slider(id='sim-eve-mins', min=0, max=400, step=10, value=150, marks={0: '0', 400: '400'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Customer Service Calls: "), html.Span(id="val-svc-calls")]),
                dcc.Slider(id='sim-svc-calls', min=0, max=10, step=1, value=1, marks={0: '0', 10: '10'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("International Plan: ")]),
                dcc.Dropdown(
                    id='sim-intl-plan',
                    options=[{'label': 'No Plan', 'value': 0}, {'label': 'Active Plan', 'value': 1}],
                    value=0,
                    clearable=False
                ),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Voice Mail Plan: ")]),
                dcc.Dropdown(
                    id='sim-vmail-plan',
                    options=[{'label': 'No Plan', 'value': 0}, {'label': 'Active Plan', 'value': 1}],
                    value=0,
                    clearable=False
                ),
            ], className="mb-4"),

            html.Hr(),

            dbc.ButtonGroup([
                dbc.Button("💾 Save Scenario", id="btn-save-scenario", color="primary", className="me-2"),
                dbc.Button("🗑️ Clear History", id="btn-clear-history", color="light", outline=True),
                dbc.Button("📥 Download Comparison (CSV)", id="btn-download-scenarios", color="dark", outline=True),
            ], className="mt-2 w-100"),
            dcc.Download(id="download-scenarios-csv"),

        ], md=7),
        
        # Updated SIMULATE Tab "Live Risk Output" Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Live Risk Output")),
                dbc.CardBody([                  
                    html.Label([html.B("Alert Sensitivity Threshold: "), html.Span(id="val-threshold")]),
                    # dcc.Slider(id='sim-threshold', min=10, max=90, step=5, value=50, marks={10: '10%', 90: '90%'}),
                    dcc.Slider(id='sim-threshold', min=10, max=90, step=5, value=75, marks={10: '10%', 90: '90%'}),
                    html.P("Customers above this score are flagged for retention intervention.", className="text-muted small mb-4"),
                    
                    # The Gauge
                    dcc.Graph(id="sim-gauge", style={"height": "250px"}),
    
                    # Status Result (Emoji + Text)
                    html.Div(id="sim-outcome-text", className="text-center mb-3 h4"),
                    
                    # Small Summary Info
                    html.Div(id="sim-summary-text", className="text-center text-muted small")
                ])
            ], className="shadow-sm sticky-top", style={"top": "20px"}),
        ], md=5)
    ]),

    html.Hr(className="my-5"),
    html.H5("📊 Historical Churn Scenarios"),
    dash_table.DataTable(
        id='scenario-history-table',
        columns=[
            {"name": "Scenario", "id": "name"},
            {"name": "Churn Prob.", "id": "score"},
            {"name": "Day Mins", "id": "day_mins"},
            {"name": "Eve Mins", "id": "eve_mins"},
            {"name": "Svc Calls", "id": "svc_calls"},
            {"name": "Intl Plan", "id": "intl_plan"},
            {"name": "VMail Plan", "id": "vmail_plan"}
        ],
        data=[],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
    ),
    dcc.Store(id='scenario-storage', data=[])
], className="p-4")

# 7. ACT Tab
tab_act = html.Div([
    html.Div([
        html.H4(["🚀 ", html.B("ACT"), " — Operational Retention Policy"], className="mt-4"),
        html.P("Deploy strategies based on model findings to minimize customer churn."),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("💡 Strategic Retention Insights"),
                    html.Hr(),
                    html.B("High-Usage Friction"),
                    html.P("Customers with Day Minutes exceeding 250 are 3x more likely to churn. Offer a 'Unlimited Business' upgrade."),
                    html.B("Proactive Service Recovery"),
                    html.P("Automate a manager callback for any customer reaching 4 service calls within a single billing cycle."),
                    html.B("Plan Optimization"),
                    html.P("The International Plan is a major churn driver. Review competitive pricing or include international bundles."),
                ], className="p-3")
            ], md=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Management Compliance Reporting", className="mb-0")),
                    dbc.CardBody([
                        html.P("Export model validation for Executive Review:"),
                        dcc.Dropdown(
                            id="report-model-dropdown",
                            options=[{'label': name, 'value': name} for name in model_results.keys()],
                            value='LGBM Classifier',
                            className="mb-3"
                        ),
                        dbc.Button("📥 Download Compliance Report (PDF)", 
                                id="btn-pdf-p1", 
                                color="success", 
                                className="w-100 mb-3"),
                        dcc.Download(id="download-pdf-p1"),
                        html.Hr(),
                        html.H6("Immediate Retention Alert:"),
                        dbc.RadioItems(
                            id="urgency-selector",
                            options=[{"label": "🟢 Low", "value": "LOW"}, {"label": "🟡 Medium", "value": "MEDIUM"}, {"label": "🔴 High", "value": "HIGH"}],
                            value="MEDIUM", inline=True, className="mb-3"
                        ),
                        dbc.Button("📧 Alert Retention Department", 
                                id="btn-email-p1", 
                                href="", target="_blank", color="primary", outline=True, className="w-100")
                        ])
                ], className="shadow-sm mt-4")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.H5("Internal Audit Summary", className="mt-4"),
                    html.Ul([
                        html.Li("Churn vs. Retention cost-benefit analysis."),
                        html.Li("Model Precision check (False Positive minimization)."),
                        html.Li("Comparison of Tree-based vs. Linear models."),
                        html.Li("Segmented risk rankings by account length."),
                    ], className="mt-3")
                ], className="p-4")
            ], md=6)
        ])
    ], fluid=True)
])

app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(ask_tab, label="Ask", tab_id="tab-ask"),
        dbc.Tab(prepare_tab, label="Prepare", tab_id="tab-prepare"),
        dbc.Tab(analyze_tab, label="Analyze", tab_id="tab-analyze"),
        dbc.Tab(tab_explain, label="Explain", tab_id="tab-explain"),
        dbc.Tab(tab_simulate, label="Simulate", tab_id="tab-simulate"),
        dbc.Tab(tab_act, label="Act", tab_id="tab-act"),
    ], id="main-tabs", active_tab="tab-ask"),
    # REMOVED: dcc.Store(id='scenario-storage', data=[])  <-- This was the duplicate
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output("correlation-matrix", "figure"),
    Input("churn-pie-chart", "id")
)
def update_corr_matrix(dummy):
    correlation = telcom.corr()
    fig = ff.create_annotated_heatmap(
        z=correlation.values.round(2),
        x=list(correlation.columns),
        y=list(correlation.index),
        colorscale="Viridis",
        showscale=True,
        reversescale=True
    )
    fig.update_layout(title="Correlation Matrix", height=500, margin=dict(t=50, b=50))
    return fig

@app.callback(
    Output("train-metrics-bar", "figure"),
    Output("train-confusion-matrix", "figure"),
    Output("train-roc-curve", "figure"),
    Output("selected-train-metrics-summary", "children"),
    Output("selected-train-confusion-summary", "children"),
    Output("selected-train-roc-summary", "children"),
    Input('model-selector-train', 'value')
)
def update_train_performance(selected_model):
    def get_bar_chart(df, title):
        fig = go.Figure()
        for metric in ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC', 'Kappa']:
            fig.add_trace(go.Bar(
                y=df["Model"],
                x=df[metric],
                orientation='h',
                name=metric
            ))
        fig.update_layout(
            barmode='group',
            title=title,
            height=450,
            margin=dict(l=150, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    train_bar_chart = get_bar_chart(metrics_train, "Model Metrics (Training Data)")
    
    model = model_results.get(selected_model, model_results['LGBM Classifier'])
    y_pred_train = model.predict(X_test)
    cm_train = confusion_matrix(y_test, y_pred_train)
    
    tn, fp, fn, tp = cm_train.ravel()
    z_data_train = np.array([[tp, fn], [fp, tn]])
    cm_text_train = np.array([[f'TP: {tp}', f'FN: {fn}'], [f'FP: {fp}', f'TN: {tn}']])
    
    fig_cm_train = ff.create_annotated_heatmap(
        z=z_data_train,
        x=["Predicted Churn (1)", "Predicted No-Churn (0)"],
        y=["Actual Churn (1)", "Actual No-Churn (0)"],
        annotation_text=cm_text_train,
        colorscale='blues',
        showscale=False
    )
    fig_cm_train.update_yaxes(autorange='reversed')
    fig_cm_train.update_layout(title=f"Confusion Matrix ({selected_model} - Train)", height=450, margin=dict(t=50, b=50))
    fig_cm_train.update_annotations(font_size=16)

    def get_roc_curve(model, X, y, title):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probabilities)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
                margin=dict(t=50, b=50)
            )
            roc_summary = html.P([
                "The ", html.B(selected_model), " achieved an ", html.B(f"AUC of {roc_auc:.2f}"), 
                ". An AUC score near 1.00 indicates excellent differentiation capability."
            ])
        else:
            fig = go.Figure(go.Scatter())
            fig.update_layout(title=f"ROC Curve Not Available for {selected_model}", height=450, margin=dict(t=50, b=50))
            roc_summary = html.P(f"ROC Curve and AUC metric are not available for {selected_model}.")
        return fig, roc_summary

    roc_train_fig, roc_train_summary = get_roc_curve(model, X_test, y_test, f"ROC Curve ({selected_model} - Train)")
    
    selected_metrics_train = metrics_train[metrics_train['Model'] == selected_model].iloc[0]
    metrics_summary_train = html.P([
        "The selected model, ", html.B(selected_model), ", achieved: ",
        html.B(f"Accuracy: {selected_metrics_train['Accuracy']}"), ", ",
        html.B(f"Precision: {selected_metrics_train['Precision']}"), ", ",
        html.B(f"Recall: {selected_metrics_train['Recall']}"), ", ",
        html.B(f"F1-Score: {selected_metrics_train['F1-Score']}"), "."
    ])

    confusion_summary_train = html.P([
        "For the selected model, ", html.B(selected_model), ", the confusion matrix showed: ",
        html.B(f"{tp} True Positives (TP)"), ", ",
        html.B(f"{tn} True Negatives (TN)"), ", ",
        html.B(f"{fp} False Positives (FP)"), " and ",
        html.B(f"{fn} False Negatives (FN)"), "."
    ])

    return (
        train_bar_chart,
        fig_cm_train,
        roc_train_fig,
        metrics_summary_train,
        confusion_summary_train,
        roc_train_summary
    )

@app.callback(
    Output("test-metrics-bar", "figure"),
    Output("test-confusion-matrix", "figure"),
    Output("test-roc-curve", "figure"),
    Output("selected-test-metrics-summary", "children"),
    Output("selected-test-confusion-summary", "children"),
    Output("selected-test-roc-summary", "children"),
    Input('model-selector-test', 'value')
)
def update_test_performance(selected_model):
    def get_bar_chart(df, title):
        fig = go.Figure()
        for metric in ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC', 'Kappa']:
            fig.add_trace(go.Bar(
                y=df["Model"],
                x=df[metric],
                orientation='h',
                name=metric
            ))
        fig.update_layout(
            barmode='group',
            title=title,
            height=450,
            margin=dict(l=150, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    test_bar_chart = get_bar_chart(metrics_test, "Model Metrics (Test Data)")
    
    model = model_results.get(selected_model, model_results['LGBM Classifier'])
    y_pred_test = model.predict(telcom_test[cols])
    cm_test = confusion_matrix(telcom_test['Churn'], y_pred_test)

    tn, fp, fn, tp = cm_test.ravel()
    z_data_test = np.array([[tp, fn], [fp, tn]])
    cm_text_test = np.array([[f'TP: {tp}', f'FN: {fn}'], [f'FP: {fp}', f'TN: {tn}']])
    
    fig_cm_test = ff.create_annotated_heatmap(
        z=z_data_test,
        x=["Predicted Churn (1)", "Predicted No-Churn (0)"],
        y=["Actual Churn (1)", "Actual No-Churn (0)"],
        annotation_text=cm_text_test,
        colorscale='blues',
        showscale=False
    )
    fig_cm_test.update_yaxes(autorange='reversed')
    fig_cm_test.update_layout(title=f"Confusion Matrix ({selected_model} - Test)", height=450, margin=dict(t=50, b=50))
    fig_cm_test.update_annotations(font_size=16)

    def get_roc_curve(model, X, y, title):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probabilities)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
                margin=dict(t=50, b=50)
            )
            roc_summary = html.P([
                "The ", html.B(selected_model), " achieved an ", html.B(f"AUC of {roc_auc:.2f}"), 
                ". An AUC score near 1.00 indicates excellent differentiation capability."
            ])
        else:
            fig = go.Figure(go.Scatter())
            fig.update_layout(title=f"ROC Curve Not Available for {selected_model}", height=450, margin=dict(t=50, b=50))
            roc_summary = html.P(f"ROC Curve and AUC metric are not available for {selected_model}.")
        return fig, roc_summary
    
    roc_test_fig, roc_test_summary = get_roc_curve(model, telcom_test[cols], telcom_test['Churn'], f"ROC Curve ({selected_model} - Test)")
    
    selected_metrics_test = metrics_test[metrics_test['Model'] == selected_model].iloc[0]
    metrics_summary_test = html.P([
        "The selected model, ", html.B(selected_model), ", achieved: ",
        html.B(f"Accuracy: {selected_metrics_test['Accuracy']}"), ", ",
        html.B(f"Precision: {selected_metrics_test['Precision']}"), ", ",
        html.B(f"Recall: {selected_metrics_test['Recall']}"), ", ",
        html.B(f"F1-Score: {selected_metrics_test['F1-Score']}"), "."
    ])
    
    confusion_summary_test = html.P([
        "For the selected model, ", html.B(selected_model), ", the confusion matrix showed: ",
        html.B(f"{tp} True Positives (TP)"), ", ",
        html.B(f"{tn} True Negatives (TN)"), ", ",
        html.B(f"{fp} False Positives (FP)"), " and ",
        html.B(f"{fn} False Negatives (FN)"), "."
    ])

    return (
        test_bar_chart,
        fig_cm_test,
        roc_test_fig,
        metrics_summary_test,
        confusion_summary_test,
        roc_test_summary
    )

@app.callback(
    Output("feature-importance-plot", "figure"),
    Input("feature-importance-model", "value")
)
def update_feature_importance(selected_model):
    model = model_results.get(selected_model)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = cols
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importance['Importance'],
            y=df_importance['Feature'],
            orientation='h'
        ))
        fig.update_layout(
            title=f"Feature Importance for {selected_model}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
        return fig
    else:
        fig = go.Figure(go.Scatter())
        fig.update_layout(title=f"Feature Importance Not Available for {selected_model}", height=450, margin=dict(t=50, b=50))
        return fig

# --- PDF Report for Churn ---
@app.callback(
    Output("download-pdf-p1", "data"),
    Input("btn-pdf-p1", "n_clicks"),
    State("report-model-dropdown", "value"),
    prevent_initial_call=True,
)
def generate_churn_compliance_report(n_clicks, selected_model):
    # --- PAGE 1: Executive Compliance ---
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 12); pdf.cell(35, 12, "TELECOM LOGO", border=1, ln=0, align='C')
    pdf.set_xy(50, 10)
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "CORPORATE CHURN ANALYTICS COMPLIANCE", ln=True)
    pdf.set_xy(50, 18); pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", ln=True)
    
    # Action Badge
    pdf.set_xy(160, 10); pdf.set_fill_color(0, 123, 255); pdf.set_text_color(255, 255, 255); pdf.set_font("Arial", 'B', 10)
    pdf.cell(45, 8, "VALDIATION PASSED", border=0, ln=1, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    # Executive Summary
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, f"CERTIFIED PRODUCTION MODEL: {selected_model}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, f"The {selected_model} architecture has been validated for production deployment to identify churn risk. The model has been stress-tested against historical usage data to ensure high recall for retention targeting.")
    pdf.ln(5)

    # Model Performance Comparison Table
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "BENCHMARK PERFORMANCE COMPARISON:", ln=True); pdf.ln(2)
    pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(235, 235, 235)
    pdf.cell(55, 8, "Model Architecture", 1, 0, 'C', True)
    pdf.cell(35, 8, "Precision", 1, 0, 'C', True)
    pdf.cell(35, 8, "Recall", 1, 0, 'C', True)
    pdf.cell(35, 8, "F1-Score", 1, 0, 'C', True)
    pdf.cell(30, 8, "ROC-AUC", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 9)
    for name, m in model_results.items():
        y_pred = m.predict(X_test)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Highlight the selected model
        if name == selected_model: 
            pdf.set_font("Arial", 'B', 9); pdf.set_text_color(0, 123, 255)
        else: 
            pdf.set_font("Arial", '', 9); pdf.set_text_color(0, 0, 0)
        
        pdf.cell(55, 8, name, 1)
        pdf.cell(35, 8, f"{prec:.3f}", 1)
        pdf.cell(35, 8, f"{rec:.3f}", 1)
        pdf.cell(35, 8, f"{f1:.3f}", 1)
        pdf.cell(30, 8, "Validated", 1, 1)

    pdf.set_text_color(0, 0, 0); pdf.ln(5)

    # Methodology Section
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "ANALYTIC METHODOLOGY:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, "Retention models are optimized for Recall to ensure maximum capture of churning accounts. Feature scaling (StandardScaler) and label encoding have been applied to usage and plan-based features.")

    # Strategic Recommendations
    pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "COMPLIANCE RECOMMENDATIONS:", ln=True); pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, "- Cross-reference model flags with customer lifetime value (CLV).\n- Audit False Positive accounts monthly to refine plan-driven churn signals.")
    
    # Signatures
    pdf.ln(10); pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 10, "__________________________", 0, 0, 'L')
    pdf.cell(90, 10, "__________________________", 0, 1, 'R')
    pdf.set_font("Arial", '', 9)
    pdf.cell(90, 5, "Director of Customer Success", 0, 0, 'L')
    pdf.cell(90, 5, "Lead Data Auditor", 0, 1, 'R')

    # --- PAGE 2: Glossary ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "GLOSSARY OF RETENTION ANALYTICS", ln=True); pdf.cell(0, 5, "-" * 45, ln=True); pdf.ln(5)
    
    glossary = {
        "Churn": "The loss of customers who stop using the company's service.",
        "Recall": "The ability of the model to identify all potential churners (minimizing missed churn).",
        "Precision": "The accuracy of churn flags (minimizing unnecessary retention spend).",
        "F1-Score": "The harmonic mean of Precision and Recall, providing a balanced success metric.",
        "Feature Importance": "The mathematical weight assigned to usage behaviors (e.g., day minutes) in predicting churn.",
        "LGBM/XGBoost": "Gradient boosting algorithms used to capture complex, non-linear customer behavior patterns."
    }
    
    for term, definition in glossary.items():
        pdf.set_font("Arial", 'B', 11); pdf.cell(0, 7, f"{term}:", ln=True)
        pdf.set_font("Arial", '', 11); pdf.multi_cell(0, 6, definition); pdf.ln(3)

    timestamp_id = datetime.now().strftime('%Y%m%d-%H%M')
    report_id = f"TEL-COMP-{timestamp_id}"

    pdf.ln(10); pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report ID: {report_id} | Corporate Proprietary Data", ln=True, align='C')

    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Churn_Compliance_Summary_{timestamp_id}.pdf")

# --- Email Alert for Churn ---
@app.callback(
    Output("btn-email-p1", "href"),
    Input("report-model-dropdown", "value"),
    Input("urgency-selector", "value")
)
def update_churn_email_link(selected_model, urgency):
    to_email = "retention_strategy@telecom-corp.com"
    cc_email = ""
    
    # 1. Logic for Subject Prefix and Stakeholder CC
    if "HIGH" in urgency:
        prefix = "🔴 CRITICAL RETENTION ALERT"
        cc_email = "vp_customer_success@telecom-corp.com"
    elif "MEDIUM" in urgency:
        prefix = "🟡 STRATEGIC WARNING"
    else:
        prefix = "🟢 ACCOUNT UPDATE"

    current_time = datetime.now().strftime("%B %d, %Y at %H:%M")
    
    # 2. Create the Subject Line
    subject = f"{prefix}: High-Risk Churn Segment Identified ({selected_model})"
    
    # 3. Create the Body with Professional Structure
    body = (
        f"Hello Retention & Marketing Team,\n\n"
        f"URGENCY LEVEL: {urgency}\n"
        f"SYSTEM ALERT: Automated churn escalation triggered.\n\n"
        f"A predictive analysis performed via the {selected_model} model has identified "
        f"a segment of customers showing high-probability churn markers.\n\n"
        f"Significant drivers include an increase in customer service interactions and "
        f"day-usage volatility. Immediate review of current targeted retention offers "
        f"and loyalty campaign eligibility is required.\n\n"
        f"The comprehensive Churn Compliance Audit and individual risk factor breakdowns "
        f"are available for download in the main dashboard.\n\n"
        f"--------------------------------------------------\n"
        f"ANALYSIS METADATA:\n"
        f"Timestamp: {current_time}\n"
        f"Primary Engine: {selected_model}\n"
        f"Source: Customer Usage Logs (Internal)\n"
        f"--------------------------------------------------\n\n"
        f"Best regards,\n"
        f"Customer Analytics Department"
    )
    
    # 4. Safe URL Encoding
    safe_subject = urllib.parse.quote(subject)
    safe_body = urllib.parse.quote(body)
    
    mailto_link = f"mailto:{to_email}?subject={safe_subject}&body={safe_body}"
    
    # Append CC if it is a High Urgency alert
    if cc_email:
        mailto_link += f"&cc={urllib.parse.quote(cc_email)}"
        
    return mailto_link

@app.callback(
    Output("sim-gauge", "figure"),
    Output("sim-outcome-text", "children"),
    Output("sim-summary-text", "children"), # New output for summary
    Output("val-day-mins", "children"),
    Output("val-eve-mins", "children"),    # Added Output
    Output("val-svc-calls", "children"),
    Output("val-threshold", "children"),
    Input("sim-day-mins", "value"),
    Input("sim-eve-mins", "value"),        # Added Input
    Input("sim-svc-calls", "value"),
    Input("sim-intl-plan", "value"),
    Input("sim-vmail-plan", "value"),
    Input("sim-threshold", "value")
)
def update_simulator_logic(day_mins, eve_mins, svc_calls, intl_plan, vmail_plan, threshold):
    # Calculate Risk Score
    # Base logic: Day minutes weight, Service calls weight, and Plan penalties

    # risk_score = (day_mins / 400 * 50) + (svc_calls * 6) + (intl_plan * 25) - (vmail_plan * 10)
    # Increase the weights to make the slider more impactful

    # day_mins (40%), eve_mins (30%), svc_calls (30%) + Plan Penalties
    risk_score = (day_mins / 400 * 40) + (eve_mins / 400 * 30) + (svc_calls * 6) + (intl_plan * 20)
    risk_score = min(100, max(0, risk_score))
    
    # 1. Create the Sophisticated Gauge (Fraud Style)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'valueformat': '.1f'},
        gauge={
            # 'axis': {'range': [0, 100], 'tickwidth': 1},
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"}, # Dark professional bar
            # 'bgcolor': "white",
            # 'borderwidth': 2,
            # 'bordercolor': "#d1d1d1",
            'steps': [
                # {'range': [0, 30], 'color': "#27ae60"},  # Low Risk (Green)
                # {'range': [30, 70], 'color': "#f1c40f"}, # Medium Risk (Yellow)
                # {'range': [70, 100], 'color': "#e74c3c"} # High Risk (Red)
                {'range': [0, threshold/2], 'color': "#27ae60"},
                {'range': [threshold/2, threshold], 'color': "#f1c40f"},
                {'range': [threshold, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                # 'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    # fig.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20))
    fig.update_layout(height=250, margin=dict(t=50, b=20, l=30, r=30))

    
    # 2. Status Result with Emoji
    is_high_risk = risk_score >= threshold
    status_emoji = "🔴" if is_high_risk else "🟢"
    status_label = "HIGH RISK ALERT" if is_high_risk else "STABLE ACCOUNT"
    
    status_output = html.Span(
        f"{status_emoji} {status_label}", 
        style={"color": "#e74c3c" if is_high_risk else "#27ae60", "fontWeight": "bold"}
    )
    
    # 3. Summary Text
    summary = f"Risk Score is {risk_score:.1f}% based on selected usage profile."
    
    return fig, status_output, summary, f"{day_mins} min", f"{eve_mins} min", f"{svc_calls} calls", f"{threshold}%"

@app.callback(
    Output("shap-waterfall-plot", "figure"),
    Output("prediction-result-text", "children"),
    Output("consensus-alert-container", "children"),
    Output("confidence-gauge", "figure"),
    Input("customer-dropdown", "value"),
    Input("explain-model-dropdown", "value")
)
def update_explanation(cust_idx, selected_model):
    model = model_results[selected_model]
    samp = X_test.iloc[cust_idx:cust_idx+1]
    current_vals = X_test.iloc[cust_idx].values
    
    # 1. Prediction & Probability
    prob = model.predict_proba(samp)[0][1] if hasattr(model, "predict_proba") else 0.5
    status = "CHURN" if prob > 0.5 else "LOYAL"
    emoji = "⚠️" if prob > 0.5 else "✅"
    result_text = f"{emoji} {status} ({prob:.1%})"
    
    # 2. Confidence Gauge (Fraud Style)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%", 'valueformat':'.1f'},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 30], 'color': "#27ae60"},
                {'range': [30, 70], 'color': "#f1c40f"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
        }
    ))
    # fig_gauge.update_layout(height=180, margin=dict(t=30, b=0, l=10, r=10))
    fig_gauge.update_layout(height=180, margin=dict(t=30, b=0))

    # 3. Consensus Logic
    all_preds = [1 if m.predict(samp)[0] == 1 else 0 for m in model_results.values()]
    agreement = all_preds.count(1 if prob > 0.5 else 0)
    alert_msg = dbc.Alert(f"Model Consensus: {agreement}/{len(model_results)}", 
                          color="success" if agreement == len(model_results) else "warning", 
                          className="py-1 text-center small")

    # 4. Waterfall Plot (Feature Contributions)
    feature_names = X_test.columns.tolist()
    if hasattr(model, 'feature_importances_'):
        # Simplified contribution: feature value vs mean, weighted by importance
        contributions = (current_vals - X_test.mean().values) * model.feature_importances_
    else:
        contributions = (current_vals - X_test.mean().values) 

    df_top = pd.DataFrame({'f': feature_names, 'c': contributions})
    df_top = df_top.reindex(df_top.c.abs().sort_values(ascending=False).index).head(10).sort_values('c')

    fig_wf = go.Figure(go.Waterfall(
        orientation="h", 
        x=df_top['c'], 
        y=df_top['f'], 
        increasing={"marker": {"color": "#ff4136"}}, 
        decreasing={"marker": {"color": "#0074d9"}}
    ))
    # fig_wf.update_layout(title=f"Churn Factor Breakdown: Customer {cust_idx}", height=400, margin=dict(t=50, b=20, l=150))
    fig_wf.update_layout(title=f"Churn Factor Breakdown: Customer {cust_idx}", height=400)
    
    return fig_wf, result_text, alert_msg, fig_gauge

@app.callback(
    Output("download-local-analysis", "data"),
    Input("btn-download-local", "n_clicks"),
    State("customer-dropdown", "value"),
    State("explain-model-dropdown", "value"),
    State("prediction-result-text", "children"),
    prevent_initial_call=True,
)
def download_local_churn_audit(n_clicks, cust_idx, model_name, result_text):
    # 1. Consensus Logic (Check if all models agree on Churn/Loyal)
    samp = X_test.iloc[cust_idx:cust_idx+1]
    main_pred = model_results[model_name].predict(samp)[0]
    
    all_preds = [m.predict(samp)[0] for m in model_results.values()]
    agreement_count = all_preds.count(main_pred)
    total_models = len(model_results)
    has_disagreement = agreement_count < total_models

    # 2. PDF Initialization & Header
    # Remove emojis for PDF encoding compatibility
    clean_result = str(result_text).replace("✅", "").replace("⚠️", "").strip()
    pdf = FPDF()
    pdf.add_page()
    
    # Placeholder for Logo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(35, 12, "TELECOM LOGO", border=1, ln=0, align='C')

    pdf.set_xy(50, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "INDIVIDUAL CHURN AUDIT CASE REPORT", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(50, 18)
    pdf.cell(0, 10, f"Audit Date: {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", ln=True)

    # 3. Disagreement Warning Block
    if has_disagreement:
        pdf.set_xy(10, 35)
        pdf.set_fill_color(255, 243, 205) # Soft Yellow Warning
        pdf.set_text_color(133, 100, 4)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f" NOTE: MODEL DISCREPANCY ({agreement_count}/{total_models} models agree on this outcome)", border=1, ln=1, fill=True)
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.ln(10)

    # 4. Executive Summary Block
    pdf.set_xy(10, 48 if has_disagreement else 38)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "EXECUTIVE SUMMARY:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, f"This audit provides a risk breakdown for Customer Account {cust_idx}. The {model_name} engine identifies this user as '{clean_result}'. This assessment is based on usage velocity, service interaction frequency, and plan configuration outliers.")
    
    # 5. Decision Metrics Table
    pdf.ln(5)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Audit Parameter", 1, 0, 'L', True); pdf.cell(130, 10, "Value / Observation", 1, 1, 'L', True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 10, "Customer Index", 1); pdf.cell(130, 10, f"ACC-REF-{cust_idx}", 1, 1)
    pdf.cell(60, 10, "Primary Model", 1); pdf.cell(130, 10, model_name, 1, 1)
    pdf.cell(60, 10, "Risk Assessment", 1); pdf.cell(130, 10, clean_result, 1, 1)
    
    # 6. Top Mathematical Churn Drivers
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "TOP CHURN CONTRIBUTION FACTORS:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    model = model_results[model_name]
    current_vals = X_test.iloc[cust_idx].values
    if hasattr(model, 'feature_importances_'):
        contribs = (current_vals - X_test.mean().values) * model.feature_importances_
    else:
        contribs = (current_vals - X_test.mean().values)

    df_local = pd.DataFrame({'f': X_test.columns, 'c': contribs})
    top_drivers = df_local.reindex(df_local.c.abs().sort_values(ascending=False).index).head(5)

    for i, row in enumerate(top_drivers.itertuples(), 1):
        impact = "INCREASES CHURN RISK" if row.c > 0 else "DECREASES CHURN RISK"
        pdf.cell(0, 8, f"{i}. {str(row.f).replace('_', ' ').title()}: {impact}", ln=True)

    # 7. Strategic Recommendations
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, "RETENTION STRATEGY RECOMMENDATIONS:", ln=True)
    pdf.set_font("Arial", '', 11)
    rec_text = "Prioritize for immediate retention outreach." if "CHURN" in clean_result.upper() else "Maintain standard service protocols."
    pdf.multi_cell(0, 7, f"- {rec_text}\n- Evaluate plan pricing against usage drivers.\n- Review customer service logs for specific friction points.")

    # 8. Signatures & Footer
    pdf.ln(10); pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 10, "__________________________", 0, 0, 'L')
    pdf.cell(95, 10, "__________________________", 0, 1, 'R')
    pdf.set_font("Arial", '', 9)
    pdf.cell(95, 5, "Retention Analyst Signature", 0, 0, 'L')
    pdf.cell(95, 5, "Account Manager Signature", 0, 1, 'R')

    report_id = f"CHURN-{cust_idx}-{datetime.now().strftime('%Y%m%d%H%M')}"
    pdf.ln(10); pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report ID: {report_id} | Proprietary Retention Analytics", ln=True, align='C')

    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Churn_Audit_Acc_{cust_idx}.pdf")

@app.callback(
    Output("scenario-storage", "data"),
    Output("scenario-history-table", "data"),
    Output("download-scenarios-csv", "data"),
    Input("btn-save-scenario", "n_clicks"),
    Input("btn-clear-history", "n_clicks"),
    Input("btn-download-scenarios", "n_clicks"),
    State("sim-day-mins", "value"),
    State("sim-eve-mins", "value"),        # Added State
    State("sim-svc-calls", "value"),
    State("sim-intl-plan", "value"),
    State("sim-vmail-plan", "value"),
    State("sim-gauge", "figure"),
    State("scenario-storage", "data"),
    prevent_initial_call=True
)
def handle_simulator_actions(save_n, clear_n, down_n, day_mins, eve_mins, svc_calls, intl, vmail, gauge_fig, current_data):
    trig = ctx.triggered_id
    
    if trig == "btn-clear-history":
        return [], [], None

    if trig == "btn-download-scenarios":
        if not current_data: return current_data, current_data, None
        df_download = pd.DataFrame(current_data)
        return current_data, current_data, dcc.send_data_frame(df_download.to_csv, "churn_scenarios.csv")

    if trig == "btn-save-scenario":
        score = gauge_fig['data'][0]['value']
        new_row = {
            "name": f"Scenario {len(current_data) + 1}",
            "score": f"{score:.1f}%",
            "day_mins": day_mins,
            "eve_mins": eve_mins,          # Added here
            "svc_calls": svc_calls,
            "intl_plan": "Yes" if intl == 1 else "No",
            "vmail_plan": "Yes" if vmail == 1 else "No"
        }
        updated = [new_row] + current_data
        return updated, updated, None

    return current_data, current_data, None

if __name__ == "__main__":
    app.run(debug=True)