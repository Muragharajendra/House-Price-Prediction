import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction System")
st.write("Enter the house details below to predict the house price.")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude",
        value=-122.23,
        format="%.5f"
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88,
        format="%.5f"
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=0.0,
        value=41.0
    )

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=0.0,
        value=880.0
    )

with col2:
    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=0.0,
        value=129.0
    )

    population = st.number_input(
        "Population",
        min_value=0.0,
        value=322.0
    )

    households = st.number_input(
        "Households",
        min_value=0.0,
        value=126.0
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        value=8.3252
    )

ocean_proximity = st.selectbox("Ocean Proximity",
    [
        "NEAR BAY",
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "ISLAND"
    ]
)

if st.button("Predict House Price", use_container_width=True):

    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()["prediction"]

            st.success("Prediction completed successfully!")

            st.metric(
                label="Predicted House Price",
                value=f"${prediction:,.2f}"
            )

        else:
            st.error(response.json())

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to FastAPI server. "
            "Make sure FastAPI is running on port 8000."
        )

    except Exception as e:
        st.error(f"Error: {e}")