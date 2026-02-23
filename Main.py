import os
import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "pipeline.pkl")
HOUSING_CSV = os.path.join(BASE_DIR, "housing.csv")
INPUT_CSV = os.path.join(BASE_DIR, "input.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "output.csv")

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For categorical columns
    cat_pipline = Pipeline([ 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipline, num_attribs), 
        ('cat', cat_pipline, cat_attribs)
    ])

    return full_pipeline


def evaluate_models(housing_prepared, housing_labels):
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
    }

    results = []
    for name, model in models.items():
        rmses = -cross_val_score(
            model,
            housing_prepared,
            housing_labels,
            scoring="neg_root_mean_squared_error",
            cv=10,
        )
        results.append((name, rmses.mean(), rmses.std()))

    results.sort(key=lambda x: x[1])
    best_name = results[0][0]
    best_model = models[best_name]
    return best_model, results

if not os.path.exists(MODEL_FILE):
    # Lets train the model
    housing = pd.read_csv(HOUSING_CSV)

    # Create a stratified test set
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv(INPUT_CSV, index=False) 
        housing = housing.loc[train_index].drop("income_cat", axis=1)  
    
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs) 
    housing_prepared = pipeline.fit_transform(housing_features)
    
    best_model, results = evaluate_models(housing_prepared, housing_labels)
    model = best_model
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model selection (RMSE, lower is better):")
    for name, mean_rmse, std_rmse in results:
        print(f"- {name}: mean={mean_rmse:.2f}, std={std_rmse:.2f}")
    print(f"Selected model: {model.__class__.__name__}")
    print("Model is trained. Congrats!")
else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(INPUT_CSV)
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv(OUTPUT_CSV, index=False)
    print("Inference is complete, results saved to output.csv Enjoy!")
