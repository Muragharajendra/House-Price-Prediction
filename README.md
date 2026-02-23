# House Price Prediction

A machine learning project that predicts median house values in California using various housing features. This project implements a Random Forest Regressor trained on the California Housing dataset to provide accurate price predictions.

## Features

- **Data Preprocessing**: Automated pipeline for handling numerical and categorical features
- **Model Training**: Random Forest Regressor with stratified sampling
- **Inference**: Batch prediction capability for new housing data
- **Model Persistence**: Saves trained model and preprocessing pipeline for reuse

## Dataset

The project uses the California Housing dataset which includes features such as:
- Median income
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Latitude/Longitude
- Ocean proximity

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-prediction
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

### Training the Model

Run the main script to train the model (if not already trained):
```bash
python Main.py
```

This will:
- Load the housing data
- Create a stratified test set
- Train a Random Forest Regressor
- Save the model and preprocessing pipeline

### Making Predictions

For inference on new data:
```bash
python Main.py
```

The script will automatically detect if the model is already trained and perform inference on `input.csv`, saving results to `output.csv`.

### Run the API Locally

1. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

2. Ensure `model.pkl` and `pipeline.pkl` exist (run `Main.py` if needed).

3. Start the API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. Send a prediction request:
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '[{\"longitude\":-122.23,\"latitude\":37.88,\"housing_median_age\":41.0,\"total_rooms\":880.0,\"total_bedrooms\":129.0,\"population\":322.0,\"households\":126.0,\"median_income\":8.3252,\"ocean_proximity\":\"NEAR BAY\"}]'
```

### Deploy on Render

This repo includes `render.yaml` and `Procfile` for Render deployment. Create a new Render Web Service and point it to this repository. The build/start commands are already defined in `render.yaml`.

## Project Structure

```
├── Main.py              # Main training and inference script
├── app.py               # FastAPI inference API
├── housing.csv          # Training dataset
├── input.csv            # Input data for predictions
├── output.csv           # Prediction results
├── model.pkl            # Trained Random Forest model
├── pipeline.pkl         # Preprocessing pipeline
├── requirements.txt     # API/runtime dependencies
├── render.yaml          # Render deployment config
├── Procfile             # Render/Heroku process file
└── README.md            # Project documentation
```

## Model Details

- **Algorithm**: Random Forest Regressor
- **Model Selection**: Linear Regression, Decision Tree, and Random Forest evaluated with RMSE cross-validation; best model is trained and saved.
- **Preprocessing**:
  - Numerical features: Median imputation + Standard scaling
  - Categorical features: One-hot encoding
- **Evaluation**: Root Mean Squared Error (RMSE)

## Dependencies

- pandas
- numpy
- scikit-learn
- joblib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
