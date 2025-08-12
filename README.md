# Titanic Survival Prediction API 

This project is a machine learning pipeline that predicts passenger survival on the Titanic using an ensemble of models (Random Forest, Logistic Regression, Gradient Boosting). It includes threshold tuning, evaluation, and a FastAPI deployment for real-time and batch predictions.

# Note

Data cleansingin this project is intentionally minimal as the goal of this was to produce a decent baseline model that can be used to demontstrate how this could be integrated into a full machine learning pipeline using an API

##  Features

- Ensemble model using VotingClassifier with:
  - Random Forest 
  - Logistic Regression 
  - Gradient Boosting
- Custom threshold tuning per model to optimise F1 score
- Model evaluation with:
  - Precision, Recall, F1, AUC
  - Confusion Matrices
- FastAPI backend for:
  - Real-time prediction
  - Batch prediction
- Auto-saves model and metadata ('model', 'threshold', 'features';) for deployment

##  How to Train the Model

1. Place 'train.csv' into the 'data/' folder.
2. Run the training script:

This will:
- Train base models
- Tune thresholds per model
- Evaluate performance
- Build an ensemble
- Save the final model and threshold into 'saved_models/final_model_bundle.pkl'

If any other model outperforms the ensemble model you can change the model in the bundle
This is done in the 'Save Model' section

model_bundle = {
    "model": ensemble_model,
    "threshold": threshold,
    "features": X.columns.tolist()
}

replace ensemble_model with the preferred model

## How to Run the API

1. Install dependencies:

pip install fastapi uvicorn scikit-learn pandas matplotlib seaborn

2. Start the API locally:

uvicorn app.main:app --reload

3. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to interact with the API using Swagger UI.

## Example API Usage

### Single Prediction
**POST** '/predict'
{
  "Pclass": 1,
  "Sex": 1,
  "Age": 29.0,
  "Fare": 120.0
}

### Batch Prediction
**POST** '/predict-batch'
{
  "passengers": [
    {
      "Pclass": 1,
      "Sex": 1,
      "Age": 29.0,
      "Fare": 120.0
    },
    {
      "Pclass": 3,
      "Sex": 0,
      "Age": 22.0,
      "Fare": 7.25
    }
  ]
}

## Author
Built by Nick Lawler  
Exploring real-world machine learning deployment and pipeline