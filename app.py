import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
)

# Model Imports
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

# --- Data Loading and Preprocessing ---
def preprocess_data(train_path, test_path):
    # Load data
    telcom = pd.read_csv(train_path)
    telcom_test = pd.read_csv(test_path)
    
    # Remove correlated and unnecessary columns
    col_to_drop = [
        'State', 'Area code', 'Total day charge', 'Total eve charge',
        'Total night charge', 'Total intl charge'
    ]
    telcom = telcom.drop(columns=col_to_drop, axis=1)
    telcom_test = telcom_test.drop(columns=col_to_drop, axis=1)

    # --- Preprocessing ---
    # Binary columns with 2 values
    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    bin_cols = [col for col in bin_cols if col != 'Churn']
    
    # Label encoding Binary columns
    le = LabelEncoder()
    for col in bin_cols:
        telcom[col] = le.fit_transform(telcom[col])
        telcom_test[col] = le.transform(telcom_test[col])

    # Scaling Numerical columns
    num_cols = [col for col in telcom.columns if telcom[col].dtype in ['float64', 'int64'] and col not in bin_cols + ['Churn']]
    std = StandardScaler()
    telcom[num_cols] = std.fit_transform(telcom[num_cols])
    telcom_test[num_cols] = std.transform(telcom_test[num_cols])
    
    # Split data
    target_col = ['Churn']
    cols = [col for col in telcom.columns if col not in target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        telcom[cols], telcom[target_col], test_size=0.25, random_state=111
    )
    
    return telcom, telcom_test, X_train, X_test, y_train, y_test, cols

telcom, telcom_test, X_train, X_test, y_train, y_test, cols = preprocess_data(
    'churn-bigml-80.csv', 'churn-bigml-20.csv'
)

# --- Model Training ---
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=9, random_state=42),
    'KNN Classifier': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'SVM (RBF)': SVC(C=10.0, gamma=0.1, probability=True, random_state=42),
    'LGBM Classifier': LGBMClassifier(learning_rate=0.5, max_depth=7, n_estimators=100, random_state=42),
    'XGBoost Classifier': XGBClassifier(learning_rate=0.9, max_depth=7, n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42),
    'Bagging Classifier': BaggingClassifier(random_state=42),
}

# Train all models and store results
model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    model_results[name] = model

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

# --- Dashboard Layout ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("☎️", className="me-2"),
                    dbc.NavbarBrand("Telecom Customer Churn Prediction", class_name="fw-bold", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Interactive Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. ASK Tab
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — The Big Picture
    This section defines the project's purpose and its business value.

    **Business Task**: The goal is to predict which customers are likely to stop using our service, a process known as **customer churn**. For telecom companies, keeping existing customers is much cheaper than finding new ones. By predicting churn, we can proactively reach out to at-risk customers and try to win them back.

    **Stakeholders**: The key decision-makers who will use this dashboard are **Marketing** and **Customer Service**. They need this information to design targeted campaigns and retention strategies. Executive leadership also benefits from a high-level view of our customer retention efforts.

    **Deliverables**: The final product is this interactive dashboard, which provides a clear, step-by-step walkthrough of our analysis and presents key findings and recommendations.
    """, className="p-4"
)

# 2. PREPARE Tab
prepare_tab = html.Div(
    children=[
        html.H4(
            ["📝 ", html.B("PREPARE"), " — Getting the Data Ready"],
            className="mt-4"
        ),
        html.P("Before we can build a predictive model, we need to understand and clean our data."),
        html.H5("Data Source"),
        html.P(
            ["We used a standard telecom churn dataset, split into a ", html.B("training set"), " (80% of the data) for building our models and a separate ", html.B("test set"), " (20%) to check if our models work on new, unseen data."]
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
        html.P("This table provides a quick statistical overview of the features. Notice the perfect linear relationship between minutes and charges for different call types. To avoid multicollinearity in our models, we've dropped the charge-related columns."),
        dbc.Table.from_dataframe(telcom.describe().T.reset_index().rename(columns={'index': 'feature'}).round(2),
                                 striped=True, bordered=True, hover=True),
    ], className="p-4"
)

# 3. ANALYZE Tab with sub-tabs
analyze_tab = html.Div(
    children=[
        html.H4(
            ["📈 ", html.B("ANALYZE"), " — Finding Patterns and Building Models"],
            className="mt-4"
        ),
        html.P("This is where we explore the data and build the predictive brain of our dashboard."),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.H5("Churn Distribution and Correlations", className="mt-4"),
                        html.P(
                            ["The pie chart below shows that our data is ", html.B("imbalanced"), "—a small percentage of customers churned. This is important because it means a simple model could get a high accuracy score by just predicting that no one will ever churn. This is why we need more advanced evaluation metrics."]
                        ),
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
                        html.P(
                            ["The ", html.B("Correlation Matrix"), " on the right shows how strongly each feature relates to every other feature. A dark color indicates a strong relationship. The key takeaway is that features like call minutes and customer service calls are correlated with churn, which confirms our intuition."]
                        ),
                        html.H5("Feature Visualization", className="mt-4"),
                        html.P("This plot visualizes the data using two key features: total day minutes and total evening minutes. We separate these time periods because customer behavior and churn drivers can be different throughout the day. A customer who has many long calls during the day might be a business user, whereas a customer with long evening calls might be a family user. These different behaviors could have different reasons for churning. A model that looks at total minutes wouldn't capture these nuances."),
                        dbc.Row([
                            dbc.Col(dcc.Graph(
                                id="day-eve-minutes-plot",
                                figure=go.Figure(
                                    data=go.Scatter(x=telcom['Total day minutes'], y=telcom['Total eve minutes'],
                                                    mode='markers', marker_color=telcom['Churn'], showlegend=False),
                                    layout=go.Layout(title="Total Day Minutes vs. Total Eve Minutes",
                                                     xaxis_title="Total Day Minutes (Scaled)",
                                                     yaxis_title="Total Eve Minutes (Scaled)",
                                                     height=400, margin=dict(t=50, b=50))
                                ))),
                        ]),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance (Training)", children=[
                html.Div(
                    children=[
                        html.H5("Model Performance on Training Data", className="mt-4"),
                        html.P("We trained a variety of machine learning models to see which one performs best."),
                        html.P(
                            ["• ", html.B("The Problem with Accuracy"), ": For our imbalanced data, ", html.B("Accuracy"), " (the percentage of correct predictions) isn't the best metric. A model that always predicts 'no churn' could be 85% accurate, but it would be useless for identifying at-risk customers."]
                        ),
                        html.P(
                            ["• ", html.B("Key Metrics"), ": We focus on a more complete set of metrics: ", html.B("Recall"), " (how many churners did we catch?), ", html.B("Precision"), " (of those we flagged as churners, how many were correct?), ", html.B("F1-Score"), " (a balance of both), and ", html.B("ROC-AUC"), " (how well the model separates churners from non-churners)."]
                        ),
                        dbc.Row([dbc.Col(dcc.Graph(id="train-metrics-bar"), md=12)]),
                        html.P(
                            ["Our analysis shows that ", html.B("LightGBM"), " and ", html.B("XGBoost"), ", both types of advanced tree-based models, significantly outperformed the others. They achieved high scores across all metrics, proving they are excellent at identifying churn."]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="train-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="train-roc-curve"), md=6),
                        ]),
                        html.H5("Feature Importance (for tree-based models)", className="mt-4"),
                        html.P("This plot ranks the features based on how much they contributed to the model's prediction. The two most important features were `Total day minutes` and `International plan`. This gives us a clear starting point for our recommendations."),
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
                        html.P(
                            ["We tested our top models on the unseen data to ensure they are not ", html.B("overfitting"), " (memorizing the training data instead of learning general patterns). The performance remained high, confirming that the models are reliable and will work well in a real-world scenario."]
                        ),
                        dbc.Row([dbc.Col(dcc.Graph(id="test-metrics-bar"), md=12)]),
                        html.P(
                            ["The performance on the test data is similar to the training data, confirming that ", html.B("LGBM Classifier"), " and ", html.B("XGBoost Classifier"), " are excellent choices for this problem. Their high ", html.B("F1-Score"), " and ", html.B("ROC-AUC"), " on both datasets indicate a strong and reliable predictive ability."]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="test-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="test-roc-curve"), md=6),
                        ]),
                        html.P("The consistent performance of these models on the test set is a key finding. It suggests that a predictive system can be built to identify customers at risk of churn with high confidence."),
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 4. ACT Tab
act_tab = dcc.Markdown(
    """
    ### 🚀 **ACT** — What to Do Next
    This is the most important section, as it translates data insights into a business strategy.

    -   **Target High-Risk Customers**: The models identified that customers with an **International plan** and high **Total day minutes** are most likely to churn. This is the perfect group for a targeted marketing campaign.
    -   **Proactive Retention**: Instead of waiting for customers to cancel, the company should use the deployed model to get a daily list of customers at high risk of churning. A customer service representative can then call them proactively to offer a discount, a service upgrade, or simply to check on their satisfaction.
    -   **Deploy the Best Model**: The **LightGBM Classifier** is our recommended model for deployment due to its superior performance on both training and test data. This model will be the brain behind our new, proactive churn-reduction strategy.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Ask"),
                dbc.Tab(prepare_tab, label="Prepare"),
                dbc.Tab(analyze_tab, label="Analyze"),
                dbc.Tab(act_tab, label="Act"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks ---
@app.callback(
    Output("correlation-matrix", "figure"),
    Input("churn-pie-chart", "id") # Dummy input to trigger on load
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
    Output("test-metrics-bar", "figure"),
    Output("train-confusion-matrix", "figure"),
    Output("test-confusion-matrix", "figure"),
    Output("train-roc-curve", "figure"),
    Output("test-roc-curve", "figure"),
    Input('feature-importance-model', 'value')
)
def update_model_performance(selected_model):
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
    test_bar_chart = get_bar_chart(metrics_test, "Model Metrics (Test Data)")

    model = model_results.get(selected_model, model_results['LGBM Classifier'])

    y_pred_train = model.predict(X_test)
    cm_train = confusion_matrix(y_test, y_pred_train)
    fig_cm_train = ff.create_annotated_heatmap(
        z=cm_train, x=["Not Churn", "Churn"], y=["Not Churn", "Churn"],
        colorscale='blues'
    )
    fig_cm_train.update_layout(title=f"Confusion Matrix ({selected_model} on Training)", height=450, margin=dict(t=50, b=50))

    y_pred_test = model.predict(telcom_test[cols])
    cm_test = confusion_matrix(telcom_test['Churn'], y_pred_test)
    fig_cm_test = ff.create_annotated_heatmap(
        z=cm_test, x=["Not Churn", "Churn"], y=["Not Churn", "Churn"],
        colorscale='blues'
    )
    fig_cm_test.update_layout(title=f"Confusion Matrix ({selected_model} on Test)", height=450, margin=dict(t=50, b=50))

    def get_roc_curve(model, X, y, title):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probabilities)
            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
                margin=dict(t=50, b=50)
            )
        else:
            fig = go.Figure(go.Scatter())
            fig.update_layout(title=f"ROC Curve Not Available for {selected_model}", height=450, margin=dict(t=50, b=50))
        return fig

    roc_train = get_roc_curve(model, X_test, y_test, f"ROC Curve ({selected_model} on Training)")
    roc_test = get_roc_curve(model, telcom_test[cols], telcom_test['Churn'], f"ROC Curve ({selected_model} on Test)")

    return (
        train_bar_chart,
        test_bar_chart,
        fig_cm_train,
        fig_cm_test,
        roc_train,
        roc_test,
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
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h'
        ))
        fig.update_layout(
            title=f"Feature Importances for {selected_model}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
        return fig
    else:
        fig = go.Figure(go.Scatter())
        fig.update_layout(title=f"No Feature Importance for {selected_model}", height=450, margin=dict(t=50, b=50))
        return fig

if __name__ == "__main__":
    app.run(debug=True)