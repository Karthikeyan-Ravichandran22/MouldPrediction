# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020
@author: win10
"""

# # Library imports
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, parse_obj_as
# import pandas as pd
# import joblib
# import os

# # Create the app object
# app = FastAPI()

# # Load the trained model, scaler, and schema
# model_path = 'trained_random_forest_classifier.pkl'
# scaler_path = 'trained_scaler.pkl'
# schema_path = 'data_schema.pkl'

# model = joblib.load(model_path)
# scaler = joblib.load(scaler_path)
# schema = joblib.load(schema_path)

# # Function to prepare and make predictions
# def prepare_and_predict(input_df, model, scaler, schema):
#     # Prepare input DataFrame for prediction
#     input_df_encoded = pd.get_dummies(input_df)
#     missing_cols = set(schema) - set(input_df_encoded.columns)
#     for c in missing_cols:
#         input_df_encoded[c] = 0
#     input_df_aligned = input_df_encoded.reindex(columns=schema, fill_value=0)

#     # Scale the features and make predictions
#     input_df_scaled = scaler.transform(input_df_aligned)
#     predictions = model.predict(input_df_scaled)
#     input_df['Predictions'] = predictions
#     return input_df

# # Define a Pydantic model for the DataFrame structure
# class DataFrameInput(BaseModel):
#     data: list
#     columns: list

# @app.post('/predict_dataframe/')
# def predict_dataframe(input: DataFrameInput):
#     try:
#         # Convert the input to a DataFrame
#         input_df = pd.DataFrame(input.data, columns=input.columns)

#         # Make predictions and get output DataFrame
#         output_df = prepare_and_predict(input_df, model, scaler, schema)
#         return output_df.to_dict(orient='records')
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Index route
# @app.get('/')
# def index():
#     return {'message': 'Welcome to the prediction service'}

# # Run the API with uvicorn
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

# -----    with csv 
# Library imports
# -*- coding: utf-8 -*-
"""
FastAPI app to handle predictions using a trained model.
"""

# Library imports
# -*- coding: utf-8 -*-
"""
FastAPI app to handle predictions using a trained model.
"""

# Library imports
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import joblib
from io import StringIO

app = FastAPI()

# Load the trained model, scaler, and schema
model_path = 'trained_random_forest_classifier.pkl'
scaler_path = 'trained_scaler.pkl'
schema_path = 'data_schema.pkl'  # Path to the schema file

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
schema = joblib.load(schema_path)  # Loading the schema

# Function to prepare and make predictions
def prepare_and_predict(input_df, model, scaler, schema):
    # Align input DataFrame with the schema
    input_df_encoded = pd.get_dummies(input_df)
    for col in set(schema) - set(input_df_encoded.columns):
        input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[schema]

    # Scale the features
    input_df_scaled = scaler.transform(input_df_encoded)

    # Predicting
    predictions = model.predict(input_df_scaled)
    return predictions

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Stream file into DataFrame
        content = await file.read()
        stream = StringIO(content.decode('utf-8'))
        input_df = pd.read_csv(stream)

        # Making predictions
        predictions = prepare_and_predict(input_df, model, scaler, schema)

        # Adding predictions to the DataFrame
        input_df['Predictions'] = predictions
        return input_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, limit_concurrency=5, limit_max_requests=100, timeout_keep_alive=120)  # Adjust the host and port as needed

# #  Run the post API
# # curl -X POST http://127.0.0.1:8000/use_existing_dataframe/
# # curl -X POST https://microserver-1dfc53516aa1.herokuapp.com/use_existing_dataframe/
