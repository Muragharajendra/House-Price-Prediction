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

1. Clone the repository (replace with your fork or target repo):
```bash
git clone https://github.com/Muragharajendra/House-Price-Prediction.git
cd House-Price-Prediction
```

2. (Recommended) create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training the Model (create model files)

If `model.pkl` and `pipeline.pkl` are not present, run the training script to create them:

```bash
python Main.py
```

This will:
- Load `housing.csv`
- Create a stratified test set and `input.csv`
- Train a Random Forest Regressor
- Save `model.pkl` and `pipeline.pkl` to the repo directory

### Running the Streamlit App

The project includes a Streamlit frontend that loads the trained `model.pkl` and `pipeline.pkl` and exposes an interactive UI. To run locally:

```bash
# after creating model artifacts (see Training section)
streamlit run frontend.py --server.port 8501
```

Enter feature values in the UI and click "Predict House Price"; the app will load `model.pkl` and `pipeline.pkl` from the repository directory and display the predicted median house value.

If `model.pkl` or `pipeline.pkl` are missing, run:

```bash
python Main.py
```
to train the model and create the artifacts.

## Docker Deployment

You can package and run both the FastAPI backend and Streamlit frontend in a single Docker container:

### 1. Build the Docker Image

```bash
docker build -t house-price-prediction .
```

### 2. Run the Docker Container

```bash
docker run -p 8501:8501 -p 8000:8000 house-price-prediction
```

Once running, you can access:
- **Streamlit Frontend**: [http://localhost:8501](http://localhost:8501)
- **FastAPI API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure (minimal files to include in repo)

```
├── app.py               # FastAPI app (serves /predict)
├── Main.py              # Training and batch inference script
├── housing.csv          # Training dataset
├── input.csv            # Input data for batch predictions
├── output.csv           # (optional) example output
├── requirements.txt     # Python dependencies
├── Dockerfile           # Optional: container image
├── .gitignore
└── README.md
```

## Model Details

- **Algorithm**: Random Forest Regressor
- **Preprocessing**:
  - Numerical features: Median imputation + Standard scaling
  - Categorical features: One-hot encoding
- **Evaluation**: Root Mean Squared Error (RMSE)

## Dependencies

- See `requirements.txt` for exact versions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Notes

- The FastAPI app expects `model.pkl` and `pipeline.pkl` in the same directory as `app.py`. Run `python Main.py` to generate them if they are not committed.

If you want me to prepare the repository for direct push to `https://github.com/Muragharajendra/House-Price-Prediction` (create a minimal commit and push), tell me and provide repository push access or credentials; otherwise follow the steps above to push.