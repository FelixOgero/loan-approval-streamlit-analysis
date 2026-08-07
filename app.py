import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Page configuration
st.set_page_config(page_title="Loan Approval Analytics Dashboard", layout="wide")

# Title
st.title("Loan Approval Prediction Dashboard")
st.markdown("### Comprehensive Analysis & Machine Learning for Loan Decisions")

# Load data with caching and column cleaning
@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset.csv')
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    # Strip spaces from string columns (like loan_status)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df

# Load the dataset
df = load_data()

# Sidebar overview
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"Total records: {df.shape[0]}")
st.sidebar.write(f"Total features: {df.shape[1]}")

# Data cleaning and preprocessing
@st.cache_data
def preprocess_data(df):
    data = df.copy()
    # Drop loan_id if present
    if 'loan_id' in data.columns:
        data.drop('loan_id', axis=1, inplace=True)
    
    # Encode categorical variables (now without spaces)
    data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
    data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Check for any missing values and drop if necessary
    if data.isnull().sum().any():
        data = data.dropna()
    return data

df_clean = preprocess_data(df)

# Separate features and target
X = df_clean.drop('loan_status', axis=1)
y = df_clean['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline for numeric features
numeric_features = X.columns.tolist()
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
@st.cache_resource
def train_models(X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
    return trained_models

trained_models = train_models(X_train, y_train)

# Evaluate models on test set
results = {}
for name, pipeline in trained_models.items():
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    }

# Select best model based on ROC AUC
best_model_name = max(results, key=lambda x: results[x]['ROC AUC'])
best_model = trained_models[best_model_name]

# EDA functions
def eda_plots(df_orig, df_clean):
    # Target distribution
    fig_target = px.histogram(df_orig, x='loan_status', title='Loan Status Distribution', color='loan_status', 
                              category_orders={'loan_status': ['Approved', 'Rejected']}, barmode='group')
    fig_target.update_layout(showlegend=False)
    
    # Correlation heatmap
    corr = df_clean.corr()
    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1))
    fig_corr.update_layout(title='Feature Correlation Matrix', height=700, width=800)
    
    # Boxplots of numeric features by loan status
    numeric_cols = X.columns
    n_cols = len(numeric_cols)
    rows = (n_cols + 1) // 2
    fig_box = make_subplots(rows=rows, cols=2, subplot_titles=numeric_cols)
    row, col = 1, 1
    for col_name in numeric_cols:
        fig_box.add_trace(go.Box(x=df_clean['loan_status'].map({1:'Approved',0:'Rejected'}), y=df_clean[col_name], name=col_name, legendgroup=col_name), row=row, col=col)
        if col == 2:
            row += 1
            col = 1
        else:
            col += 1
    fig_box.update_layout(height=400 * rows, title_text="Distribution of Numeric Features by Loan Status", showlegend=False)
    
    # CIBIL score distribution
    fig_cibil = px.histogram(df_clean, x='cibil_score', color='loan_status', barmode='overlay', 
                             title='CIBIL Score Distribution by Loan Status', labels={'loan_status':'Loan Status'})
    fig_cibil.update_layout(bargap=0.1)
    
    # Income vs Loan Amount
    fig_scatter = px.scatter(df_clean, x='income_annum', y='loan_amount', color='loan_status', 
                             title='Annual Income vs Loan Amount', opacity=0.6)
    
    return fig_target, fig_corr, fig_box, fig_cibil, fig_scatter

# User input for predictions
def user_input_features():
    st.sidebar.header("User Input Features for Prediction")
    no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    education = st.sidebar.selectbox("Education", options=['Graduate', 'Not Graduate'])
    self_employed = st.sidebar.selectbox("Self Employed", options=['Yes', 'No'])
    income_annum = st.sidebar.number_input("Annual Income (in INR)", min_value=0, value=5000000, step=100000)
    loan_amount = st.sidebar.number_input("Loan Amount (in INR)", min_value=0, value=15000000, step=100000)
    loan_term = st.sidebar.number_input("Loan Term (in years)", min_value=1, max_value=20, value=12)
    cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900, value=700)
    residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, value=5000000, step=100000)
    commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, value=5000000, step=100000)
    luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0, value=5000000, step=100000)
    bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, value=5000000, step=100000)
    
    data = {
        'no_of_dependents': no_of_dependents,
        'education': 1 if education == 'Graduate' else 0,
        'self_employed': 1 if self_employed == 'Yes' else 0,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Exploratory Analysis", "Model Performance", "Predictions", "Download Data"])

with tab1:
    st.header("Loan Application Dataset")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    st.write("Data Types and Missing Values:")
    col1, col2 = st.columns(2)
    with col1:
        st.write(df.dtypes)
    with col2:
        st.write("Missing values per column:")
        st.write(df.isnull().sum())
    st.write("Summary Statistics:")
    st.dataframe(df.describe())
    
    approval_rate = df['loan_status'].value_counts(normalize=True)['Approved'] * 100
    st.metric("Overall Loan Approval Rate", f"{approval_rate:.2f}%")

with tab2:
    st.header("Exploratory Data Analysis")
    st.subheader("Loan Status Distribution")
    fig_target, fig_corr, fig_box, fig_cibil, fig_scatter = eda_plots(df, df_clean)
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.subheader("Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Feature Distributions by Loan Status")
    st.plotly_chart(fig_box, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("CIBIL Score Analysis")
        st.plotly_chart(fig_cibil, use_container_width=True)
    with col_b:
        st.subheader("Income vs Loan Amount")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    fig_cibil_box = px.box(df_clean, x='loan_status', y='cibil_score', color='loan_status', 
                           title='CIBIL Score vs Loan Status', labels={'loan_status':'Loan Status', 'cibil_score':'CIBIL Score'})
    st.plotly_chart(fig_cibil_box, use_container_width=True)

with tab3:
    st.header("Machine Learning Model Performance")
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
    
    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"ROC AUC: {results[best_model_name]['ROC AUC']:.4f}")
    st.write(f"Accuracy: {results[best_model_name]['Accuracy']:.4f}")
    st.write(f"Precision: {results[best_model_name]['Precision']:.4f}")
    st.write(f"Recall: {results[best_model_name]['Recall']:.4f}")
    st.write(f"F1 Score: {results[best_model_name]['F1 Score']:.4f}")
    
    # Confusion Matrix
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Rejected', 'Approved'], y=['Rejected', 'Approved'])
    fig_cm.update_layout(title="Confusion Matrix - Best Model")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # ROC Curve
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{best_model_name} (AUC = {results[best_model_name]["ROC AUC"]:.3f})'))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random Classifier'))
    roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(roc_fig, use_container_width=True)
    
    # Feature Importance
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_names = X.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title=f'Feature Importance ({best_model_name})')
        st.plotly_chart(fig_fi, use_container_width=True)
    elif hasattr(best_model.named_steps['classifier'], 'coef_'):
        coef = best_model.named_steps['classifier'].coef_[0]
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef}).sort_values('Coefficient', ascending=False)
        fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', title='Logistic Regression Coefficients')
        st.plotly_chart(fig_coef, use_container_width=True)

with tab4:
    st.header("Loan Status Prediction")
    st.write("Provide the loan application details below to get an instant prediction.")
    input_df = user_input_features()
    if st.button("Predict Loan Status"):
        prediction = best_model.predict(input_df)[0]
        probability = best_model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.success(f"Loan Application: Approved with probability {probability:.2f}")
        else:
            st.error(f"Loan Application: Rejected with probability {1-probability:.2f}")
        st.write("Input Features Used:")
        st.dataframe(input_df)

with tab5:
    st.header("Download Dataset")
    st.write("Download the original loan approval dataset for offline analysis.")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(df)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='loan_approval_dataset.csv',
        mime='text/csv'
    )
    
    st.write("### Data Dictionary")
    st.markdown("""
    - **loan_id**: Unique identifier for each loan application
    - **no_of_dependents**: Number of dependents of the applicant
    - **education**: Graduate / Not Graduate
    - **self_employed**: Yes / No
    - **income_annum**: Annual income of the applicant (in INR)
    - **loan_amount**: Requested loan amount (in INR)
    - **loan_term**: Loan repayment term (in years)
    - **cibil_score**: Credit score (300-900)
    - **residential_assets_value**: Value of residential assets (in INR)
    - **commercial_assets_value**: Value of commercial assets (in INR)
    - **luxury_assets_value**: Value of luxury assets (in INR)
    - **bank_asset_value**: Value of bank assets (in INR)
    - **loan_status**: Target variable (Approved / Rejected)
    """)

# Additional insights in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Key Insights")
st.sidebar.write(f"Best performing model: **{best_model_name}**")
st.sidebar.write(f"ROC AUC: **{results[best_model_name]['ROC AUC']:.3f}**")
st.sidebar.write(f"Accuracy: **{results[best_model_name]['Accuracy']:.3f}**")
st.sidebar.write(f"Total loans analyzed: **{df.shape[0]}**")
st.sidebar.write(f"Approval rate: **{approval_rate:.1f}%**")