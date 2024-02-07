from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Paths to the trained model and scaler
model_path = 'trained_random_forest_classifier.pkl'
scaler_path = 'trained_scaler.pkl'

# Load the trained model and scaler
classifier = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define a Pydantic model for the input data structure
class PredictionInput(BaseModel):
    # Add your input features here
    # Example:
    width: float
    length: float
    floor_area: float
    window_width: float
    window_height: float
    provided_purge: float
    required_purge: float
    # Add other features as needed

@app.post("/predict")
def make_prediction(input_data: PredictionInput):
    try:
        # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Scale the features using the loaded scaler
        input_df_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = classifier.predict(input_df_scaled)

        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
