"""
streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta

from src.models import load_model
from src.feature_engineering import (
    create_date_features,
    encode_categoricals,
    load_training_columns,
    align_features,
    load_scaler,
)
from src.constants import SEASON_MAP



st.set_page_config(
    page_title="Flight Fare Predictor | Bangladesh",
    page_icon="images/plane.png" if Path("images/plane.png").exists() else "✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# Custom CSS Styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #e0e0e0;
        font-size: 1.1rem;
    }

    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }

    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8f0fe;
    }

    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }

    .prediction-amount {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    .prediction-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Flight route display */
    .route-display {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .route-cities {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e3c72;
    }

    .route-arrow {
        color: #667eea;
        margin: 0 1rem;
    }

    /* Stats cards */
    .stat-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3c72;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #666;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Input field styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }

    .stDateInput > div > div {
        border-radius: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)





# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
TRAIN_COLUMNS_PATH = PROJECT_ROOT / "data" / "processed" / "train_columns.json"
EDA_KPIS_PATH = PROJECT_ROOT / "data" / "processed" / "eda_kpis.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"

AIRLINES = [
    "Biman Bangladesh Airlines", "US-Bangla Airlines", "NovoAir", "Air Astra",
    "Emirates", "Qatar Airways", "Singapore Airlines", "Thai Airways",
    "Malaysian Airlines", "AirAsia", "IndiGo", "Air India", "Vistara",
    "British Airways", "Turkish Airlines", "Etihad Airways", "Gulf Air",
    "Cathay Pacific", "SriLankan Airlines", "Kuwait Airways", "Saudia",
    "FlyDubai", "Air Arabia", "Lufthansa"
]

SOURCES = {
    "DAC": ("Dhaka", "Hazrat Shahjalal International Airport"),
    "CGP": ("Chittagong", "Shah Amanat International Airport"),
    "CXB": ("Cox's Bazar", "Cox's Bazar Airport"),
    "ZYL": ("Sylhet", "Osmani International Airport"),
    "RJH": ("Rajshahi", "Shah Makhdum Airport"),
    "JSR": ("Jessore", "Jessore Airport"),
    "BZL": ("Barisal", "Barisal Airport"),
    "SPD": ("Saidpur", "Saidpur Airport"),
}

DESTINATIONS = {
    "DXB": ("Dubai", "UAE"),
    "SIN": ("Singapore", "Singapore"),
    "BKK": ("Bangkok", "Thailand"),
    "KUL": ("Kuala Lumpur", "Malaysia"),
    "DOH": ("Doha", "Qatar"),
    "DEL": ("Delhi", "India"),
    "CCU": ("Kolkata", "India"),
    "LHR": ("London", "UK"),
    "JFK": ("New York", "USA"),
    "YYZ": ("Toronto", "Canada"),
    "IST": ("Istanbul", "Turkey"),
    "JED": ("Jeddah", "Saudi Arabia"),
    "DAC": ("Dhaka", "Bangladesh"),
    "CGP": ("Chittagong", "Bangladesh"),
    "CXB": ("Cox's Bazar", "Bangladesh"),
    "ZYL": ("Sylhet", "Bangladesh"),
    "RJH": ("Rajshahi", "Bangladesh"),
    "JSR": ("Jessore", "Bangladesh"),
    "BZL": ("Barisal", "Bangladesh"),
    "SPD": ("Saidpur", "Bangladesh"),
}

TRAVEL_CLASSES = {
    "Economy": "Best value for money",
    "Business": "Extra comfort and services",
    "First Class": "Premium luxury experience"
}




# Cached Resource Loaders
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def get_training_columns():
    return load_training_columns(TRAIN_COLUMNS_PATH)

@st.cache_resource
def get_scaler():
    return load_scaler()

@st.cache_data
def load_eda_kpis():
    if EDA_KPIS_PATH.exists():
        with open(EDA_KPIS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_comparison():
    if MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return None




# Prediction Function
def predict_fare(airline, source, destination, travel_date, travel_class,
                 stopovers="Direct", duration=3.0, days_before=30):
   
    dt = pd.to_datetime(travel_date)
    month = dt.month
    day = dt.day
    weekday = dt.weekday()
    weekday_name = dt.day_name()
    
    

    season = SEASON_MAP[month]
    
    
    

    # Create a DataFrame with all training columns initialized to 0
    training_columns = get_training_columns()
    row = pd.DataFrame([[0] * len(training_columns)], columns=training_columns)

    # Set numeric features
    if "Duration" in row.columns:
        row["Duration"] = duration
    if "DaysBeforeDeparture" in row.columns:
        row["DaysBeforeDeparture"] = days_before
    if "Month" in row.columns:
        row["Month"] = month
    if "Day" in row.columns:
        row["Day"] = day
    if "Weekday" in row.columns:
        row["Weekday"] = weekday
        
        

    # Set one-hot encoded features to 1 where applicable
    one_hot_features = [
        f"Airline_{airline}",
        f"Source_{source}",
        f"Destination_{destination}",
        f"Stopovers_{stopovers}",
        "Aircraft Type_Boeing 737",
        f"Class_{travel_class}",
        "Booking Source_Online Website",
        "Seasonality_Regular",
        f"WeekdayName_{weekday_name}",
        f"Season_{season}",
    ]



    for col in one_hot_features:
        if col in row.columns:
            row[col] = 1

    model = get_model()
    pred_log = model.predict(row)[0]
    prediction = float(np.expm1(pred_log))

    return max(prediction, 0)





# Header
st.markdown("""
<div class="main-header">
    <h1>Flight Fare Predictor</h1>
    <p>Get instant fare estimates for flights from Bangladesh to destinations worldwide</p>
</div>
""", unsafe_allow_html=True)




# Main Layout
col_form, col_spacer, col_result = st.columns([2, 0.2, 1.5])

with col_form:
    st.markdown('<div class="card"><div class="card-header">Book Your Flight</div>', unsafe_allow_html=True)

    # Row 1: From and To
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox(
            "From",
            options=list(SOURCES.keys()),
            format_func=lambda x: f"{SOURCES[x][0]} ({x})",
            index=0
        )
    with col2:
        destination = st.selectbox(
            "To",
            options=list(DESTINATIONS.keys()),
            format_func=lambda x: f"{DESTINATIONS[x][0]} ({x})",
            index=0
        )

    # Row 2: Date and Class
    col1, col2 = st.columns(2)
    with col1:
        travel_date = st.date_input(
            "Departure Date",
            value=date.today() + timedelta(days=14),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365)
        )
    with col2:
        travel_class = st.selectbox(
            "Travel Class",
            options=list(TRAVEL_CLASSES.keys()),
            index=0
        )

    # Row 3: Airline
    airline = st.selectbox(
        "Preferred Airline",
        options=AIRLINES,
        index=0
    )

    # Advanced options in expander
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            stopovers = st.selectbox(
                "Stopovers",
                options=["Direct", "1 Stop", "2+ Stops"],
                index=0
            )
        with col2:
            duration = st.slider(
                "Est. Duration (hours)",
                min_value=0.5,
                max_value=16.0,
                value=3.0,
                step=0.5
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Calculate days before departure
    days_before = (travel_date - date.today()).days

    # Predict button
    predict_clicked = st.button("Get Fare Estimate", type="primary", use_container_width=True)

with col_result:
    # Route Display
    source_city = SOURCES[source][0]
    dest_city = DESTINATIONS[destination][0]
    dest_country = DESTINATIONS[destination][1]

    st.markdown(f"""
    <div class="route-display">
        <span class="route-cities">{source_city}</span>
        <span class="route-arrow">---></span>
        <span class="route-cities">{dest_city}</span>
        <br><small style="color: #666;">{dest_country}</small>
    </div>
    """, unsafe_allow_html=True)

    # Flight details summary
    st.markdown(f"""
    <div class="stat-card" style="margin-bottom: 1rem;">
        <div class="stat-label">Travel Date</div>
        <div class="stat-value">{travel_date.strftime('%b %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Days Until</div>
            <div class="stat-value">{days_before}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Class</div>
            <div class="stat-value">{travel_class.split()[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Prediction Result
    if predict_clicked:
        with st.spinner("Calculating best fare..."):
            try:
                predicted_fare = predict_fare(
                    airline=airline,
                    source=source,
                    destination=destination,
                    travel_date=travel_date,
                    travel_class=travel_class,
                    stopovers=stopovers,
                    duration=duration,
                    days_before=days_before,
                )

                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Estimated Fare</div>
                    <div class="prediction-amount">BDT {predicted_fare:,.0f}</div>
                    <div class="prediction-label">{airline}</div>
                </div>
                """, unsafe_allow_html=True)

                # Tips
                st.info(f"Tip: Book {days_before} days in advance for better rates!")

            except Exception as e:
                st.error(f"Unable to calculate fare: {str(e)}")
                
                
                


# Analytics Dashboard
st.markdown("---")
st.markdown("## Analytics & Insights")

kpis = load_eda_kpis()

if kpis:
    tab1, tab2, tab3 = st.tabs(["Airline Comparison", "Route Analysis", "Seasonal Trends"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            airline_data = kpis.get("avg_fare_by_airline", {})
            if airline_data:
                df = pd.DataFrame({
                    "Airline": list(airline_data.keys()),
                    "Average Fare (BDT)": list(airline_data.values())
                }).sort_values("Average Fare (BDT)", ascending=True).tail(10)

                st.bar_chart(df.set_index("Airline"))

        with col2:
            st.markdown("### Top Airlines by Price")
            if airline_data:
                sorted_airlines = sorted(airline_data.items(), key=lambda x: x[1], reverse=True)
                for i, (name, fare) in enumerate(sorted_airlines[:5], 1):
                    st.markdown(f"**{i}. {name}**")
                    st.caption(f"BDT {fare:,.0f} avg")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Most Popular Routes")
            popular = kpis.get("popular_routes", {})
            if popular:
                for route, count in list(popular.items())[:5]:
                    st.markdown(f"**{route}**")
                    st.progress(count / max(popular.values()))
                    st.caption(f"{count} flights")

        with col2:
            st.markdown("### Premium Routes")
            expensive = kpis.get("expensive_routes", {})
            if expensive:
                for route, fare in list(expensive.items())[:5]:
                    st.markdown(f"**{route}**")
                    st.caption(f"BDT {fare:,.0f} average")

    with tab3:
        seasonal = kpis.get("avg_fare_by_season", {})
        if seasonal:
            col1, col2 = st.columns([2, 1])

            with col1:
                df = pd.DataFrame({
                    "Season": list(seasonal.keys()),
                    "Average Fare": list(seasonal.values())
                })
                st.bar_chart(df.set_index("Season"))

            with col2:
                st.markdown("### Seasonal Insights")

                winter_fare = seasonal.get("Winter", 0)
                other_avg = np.mean([v for k, v in seasonal.items() if k != "Winter"])
                premium = ((winter_fare - other_avg) / other_avg) * 100 if other_avg > 0 else 0

                st.metric("Winter Premium", f"+{premium:.0f}%", delta=None)
                st.caption("Fares increase during winter holidays")

                cheapest = min(seasonal.items(), key=lambda x: x[1])
                st.metric("Best Season to Fly", cheapest[0])
                st.caption(f"Average fare: BDT {cheapest[1]:,.0f}")

else:
    st.warning("Analytics data not available. Please run the ML pipeline first.")
    
    


# Model Performance (collapsible)
with st.expander("Model Performance Metrics"):
    model_df = load_model_comparison()
    if model_df is not None:
        col1, col2, col3 = st.columns(3)
        best = model_df.iloc[0]

        # Handle different column name formats (R2 or R²)
        r2_col = "R2" if "R2" in model_df.columns else "R²"

        with col1:
            st.metric("Best Model", best["Model"].replace("_", " ").title())
        with col2:
            st.metric("Accuracy (R²)", f"{best[r2_col]:.1%}")
        with col3:
            if "MAE (BDT)" in model_df.columns:
                st.metric("Avg Error (MAE)", f"৳{best['MAE (BDT)']:,.0f} BDT")
            else:
                st.metric("Error Rate (MAE)", f"{best['MAE']:.4f}")

        fmt = {r2_col: "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}
        if "MAE (BDT)" in model_df.columns:
            fmt["MAE (BDT)"] = "{:,.0f}"
        if "RMSE (BDT)" in model_df.columns:
            fmt["RMSE (BDT)"] = "{:,.0f}"
        st.dataframe(
            model_df.style.format(fmt),
            use_container_width=True,
            hide_index=True
        )




# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <small>
            Flight Fare Prediction System | Powered by Machine Learning<br>
            Developed by Noah Jamal Nabila | © 2026
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
