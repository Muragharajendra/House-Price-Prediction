# House Price Prediction

A machine learning project that predicts median house values in California using various housing features. This project implements a Random Forest Regressor trained on the California Housing dataset to provide accurate price predictions.


## Deployed Application

<p align="center">
  <img src="https://raw.githubusercontent.com/Muragharajendra/House-Price-Prediction/master/Deployed_Snapshot.png" width="900">
</p>

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

The project includes a Streamlit frontend that can connect to the FastAPI backend or run predictions directly using the serialized models (`model.pkl` and `pipeline.pkl`).

To run locally:
```bash
# Optional: start the FastAPI backend (in one terminal)
uvicorn app:app --host 0.0.0.0 --port 8000

# Start the Streamlit frontend (in another terminal)
streamlit run frontend.py --server.port 8501
```

If the FastAPI backend is running, the Streamlit app will query it. If the backend is not running or unreachable, the Streamlit app gracefully falls back to **direct local inference** using `model.pkl` and `pipeline.pkl` in the repository directory.

If `model.pkl` or `pipeline.pkl` are missing, run:
```bash
python Main.py
```
to train the model and create the artifacts.

## Streamlit Community Cloud Deployment

To deploy this application to **Streamlit Community Cloud** (share.streamlit.io):

1. **Push your code to GitHub**: Make sure `model.pkl`, `pipeline.pkl`, `requirements.txt`, and `frontend.py` are committed and pushed to your GitHub repository (`https://github.com/Muragharajendra/House-Price-Prediction`).
2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in.
   - Click **New app**.
   - Select your repository (`Muragharajendra/House-Price-Prediction`), branch, and set the **Main file path** to `frontend.py`.
   - Click **Deploy!**

The app will install the dependencies from `requirements.txt` and launch. Since the FastAPI backend won't be running on Streamlit Cloud, the app will automatically fall back to **Direct Inference Mode** and perform the predictions directly inside the Streamlit server!

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
