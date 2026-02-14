import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border: none;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏡 California Housing Price Predictor")
st.markdown("### Predict house prices using Machine Learning")
st.markdown("---")

# Load and cache data
@st.cache_data
def load_data():
    """Load the housing dataset"""
    try:
        data = pd.read_csv("housing.csv")
        return data
    except FileNotFoundError:
        st.error("⚠️ Error: housing.csv file not found. Please make sure the file is in the same directory.")
        return None

# Train and cache model
@st.cache_resource
def train_model(data, model_type='Random Forest'):
    """Train the selected model"""
    # Prepare data
    # Handle missing values
    data = data.dropna()
    
    # Feature engineering
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    
    # Select features
    X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
              'total_bedrooms', 'population', 'households', 'median_income',
              'rooms_per_household', 'bedrooms_per_room', 'population_per_household']]
    y = data['median_house_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model based on selection
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    else:  # Decision Tree
        model = DecisionTreeRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}, X_test, y_test, y_pred

# Load data
data = load_data()

if data is not None:
    # Sidebar for model selection and settings
    st.sidebar.header("⚙️ Model Settings")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ['Random Forest', 'Linear Regression', 'Decision Tree'],
        help="Choose the machine learning model for predictions"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 About")
    st.sidebar.info(
        "This app predicts California housing prices using machine learning. "
        "Adjust the features on the main page to get predictions."
    )
    
    # Train model
    with st.spinner(f'Training {model_type} model...'):
        model, metrics, X_test, y_test, y_pred = train_model(data, model_type)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Performance", "📊 Data Insights"])
    
    # Tab 1: Prediction
    with tab1:
        st.header("Enter Housing Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Location")
            longitude = st.slider("Longitude", 
                                 float(data['longitude'].min()), 
                                 float(data['longitude'].max()), 
                                 float(data['longitude'].median()),
                                 help="Geographic longitude coordinate")
            
            latitude = st.slider("Latitude", 
                                float(data['latitude'].min()), 
                                float(data['latitude'].max()), 
                                float(data['latitude'].median()),
                                help="Geographic latitude coordinate")
            
            housing_age = st.slider("Housing Median Age (years)", 
                                   int(data['housing_median_age'].min()), 
                                   int(data['housing_median_age'].max()), 
                                   int(data['housing_median_age'].median()),
                                   help="Median age of houses in the area")
        
        with col2:
            st.subheader("Property Details")
            total_rooms = st.number_input("Total Rooms", 
                                         min_value=1, 
                                         max_value=int(data['total_rooms'].max()), 
                                         value=int(data['total_rooms'].median()),
                                         help="Total number of rooms in the area")
            
            total_bedrooms = st.number_input("Total Bedrooms", 
                                            min_value=1, 
                                            max_value=int(data['total_bedrooms'].max()), 
                                            value=int(data['total_bedrooms'].median()),
                                            help="Total number of bedrooms in the area")
            
            households = st.number_input("Households", 
                                        min_value=1, 
                                        max_value=int(data['households'].max()), 
                                        value=int(data['households'].median()),
                                        help="Number of households in the area")
        
        with col3:
            st.subheader("Demographics")
            population = st.number_input("Population", 
                                        min_value=1, 
                                        max_value=int(data['population'].max()), 
                                        value=int(data['population'].median()),
                                        help="Total population in the area")
            
            median_income = st.number_input("Median Income (tens of thousands)", 
                                           min_value=float(data['median_income'].min()), 
                                           max_value=float(data['median_income'].max()), 
                                           value=float(data['median_income'].median()),
                                           step=0.1,
                                           help="Median household income in tens of thousands of dollars")
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🎯 Predict House Price", use_container_width=True):
            # Calculate engineered features
            rooms_per_household = total_rooms / households
            bedrooms_per_room = total_bedrooms / total_rooms
            population_per_household = population / households
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'longitude': [longitude],
                'latitude': [latitude],
                'housing_median_age': [housing_age],
                'total_rooms': [total_rooms],
                'total_bedrooms': [total_bedrooms],
                'population': [population],
                'households': [households],
                'median_income': [median_income],
                'rooms_per_household': [rooms_per_household],
                'bedrooms_per_room': [bedrooms_per_room],
                'population_per_household': [population_per_household]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display prediction with styling
            st.success("### Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price", f"${prediction:,.0f}", help="Estimated median house value")
            with col2:
                st.metric("Price per Room", f"${prediction/total_rooms:,.0f}", help="Price divided by total rooms")
            with col3:
                st.metric("Price per Household", f"${prediction/households:,.0f}", help="Price divided by households")
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                st.write("**Location:**", f"({longitude:.2f}, {latitude:.2f})")
                st.write("**Property:**", f"{total_rooms} rooms, {total_bedrooms} bedrooms, {households} households")
                st.write("**Demographics:**", f"Population: {population}, Median Income: ${median_income*10000:,.0f}")
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance Metrics")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}", 
                     help="Average prediction error")
        with col2:
            st.metric("Root Mean Squared Error", f"${metrics['RMSE']:,.0f}",
                     help="Standard deviation of prediction errors")
        with col3:
            st.metric("R² Score", f"{metrics['R2']:.4f}",
                     help="Proportion of variance explained (closer to 1 is better)")
        
        st.markdown("---")
        
        # Prediction vs Actual plot
        st.subheader("Predicted vs Actual Prices")
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                size=5,
                color=y_pred,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Predicted Price")
            ),
            hovertemplate='Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            hovermode='closest',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        st.subheader("Residuals Distribution")
        residuals = y_test - y_pred
        
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name='Residuals',
            marker_color='lightblue'
        ))
        
        fig_residuals.update_layout(
            xaxis_title="Residual ($)",
            yaxis_title="Frequency",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Model comparison info
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Model Type:** {model_type}")
            st.write(f"**Training Samples:** {len(data) - len(X_test)}")
            st.write(f"**Test Samples:** {len(X_test)}")
            st.write(f"**Features Used:** 11 (including engineered features)")
    
    # Tab 3: Data Insights
    with tab3:
        st.header("Dataset Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            fig_price = px.histogram(data, x='median_house_value', 
                                    nbins=50,
                                    title="Distribution of House Prices",
                                    labels={'median_house_value': 'House Price ($)'})
            fig_price.update_traces(marker_color='lightgreen')
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.subheader("Income vs Price")
            fig_income = px.scatter(data.sample(1000), 
                                   x='median_income', 
                                   y='median_house_value',
                                   color='median_house_value',
                                   title="Median Income vs House Price",
                                   labels={'median_income': 'Median Income (10k$)', 
                                          'median_house_value': 'House Price ($)'},
                                   color_continuous_scale='Viridis')
            st.plotly_chart(fig_income, use_container_width=True)
        
        st.subheader("Geographic Price Distribution")
        fig_map = px.scatter(data.sample(2000), 
                            x='longitude', 
                            y='latitude',
                            color='median_house_value',
                            size='population',
                            title="California Housing Prices by Location",
                            labels={'median_house_value': 'House Price ($)'},
                            color_continuous_scale='RdYlGn',
                            hover_data=['median_income', 'housing_median_age'])
        fig_map.update_layout(height=600)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Dataset statistics
        with st.expander("📊 Dataset Statistics"):
            st.write(data.describe())
        
        # Show sample data
        with st.expander("🔍 View Sample Data"):
            st.dataframe(data.head(10))

else:
    st.error("Unable to load the dataset. Please check if housing.csv is in the correct location.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Data: California Housing Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)
