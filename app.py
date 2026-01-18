import numpy as np
import pickle
import pandas as pd
import gradio as gr

# Pickle model loading function

with open("svc_water_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)




# Main Logic 
def predict_water_quality(
      ph, Hardness, Solids,Chloramines,Sulfate,Conductivity
      ,Organic_carbon,Trihalomethanes,Turbidity,
):
    input_data ={
        'ph': [ph],
        'Hardness': [Hardness],
        'Solids': [Solids],
        'Chloramines': [Chloramines],
        'Sulfate': [Sulfate],
        'Conductivity': [Conductivity],
        'Organic_carbon': [Organic_carbon],
        'Trihalomethanes': [Trihalomethanes],
        'Turbidity': [Turbidity]
    }
    input_df = pd.DataFrame(input_data)
    prediction = loaded_model.predict(input_df)[0]

    mapping ={
        0: 'Not potable',
        1: 'Potable'
    } 

    return mapping[prediction]

#Interface part 
apps = gr.Interface(
    fn = predict_water_quality,
    inputs = [
        gr.Number(label="pH", value=7.0),
        gr.Number(label="Hardness", value=200),
        gr.Number(label="Solids (TDS)", value=20000),
        gr.Number(label="Chloramines", value=7),
        gr.Number(label="Sulfate", value=300),
        gr.Number(label="Conductivity", value=400),
        gr.Number(label="Organic Carbon", value=10),
        gr.Number(label="Trihalomethanes", value=60),
        gr.Number(label="Turbidity", value=4)
    ],
    outputs=gr.Textbox(label="Water Potability Prediction"),
    title="Water Quality Potability Prediction",
    description="Enter the water quality parameters to predict if the water is potable or not."
)


# Launch the app
apps.launch(share=True)