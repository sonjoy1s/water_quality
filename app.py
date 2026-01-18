import numpy as np
import pickle
import pandas as pd
import gradio as gr
from sklearn.svm import SVC

# Load trained model
with open("svc_water_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)


# Prediction function
def predict_water_quality(
    ph, hardness, solids, chloramines, sulfate,
    conductivity, organic_carbon, trihalomethanes, turbidity
):
    input_df = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity]
    })

    pred = loaded_model.predict(input_df)[0]

    return (
        "✅ **Potable Water** – Safe for drinking"
        if pred == 1
        else "❌ **Not Potable** – Not safe for drinking"
    )


# Custom CSS (Glassmorphism)
custom_css = """
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.gradio-container {
    max-width: 900px !important;
    margin: auto;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

h1, h2, h3, label {
    color: white !important;
}

footer {display: none;}
"""


# UI Layout
with gr.Blocks(css=custom_css) as app:

    gr.Markdown(
        """
        # 💧 Water Quality Potability Prediction
        ### Machine Learning Based Water Safety Analysis  
        Enter the chemical parameters of water to check if it is **safe for drinking**.
        """
    )

    with gr.Row():
        ph = gr.Number(label="pH", value=7.0)
        hardness = gr.Number(label="Hardness", value=200)

    with gr.Row():
        solids = gr.Number(label="Solids (TDS)", value=20000)
        chloramines = gr.Number(label="Chloramines", value=7)

    with gr.Row():
        sulfate = gr.Number(label="Sulfate", value=300)
        conductivity = gr.Number(label="Conductivity", value=400)

    with gr.Row():
        organic_carbon = gr.Number(label="Organic Carbon", value=10)
        trihalomethanes = gr.Number(label="Trihalomethanes", value=60)

    turbidity = gr.Number(label="Turbidity", value=4)

    predict_btn = gr.Button("🔍 Predict Water Quality", variant="primary")
    output = gr.Markdown()

    predict_btn.click(
        predict_water_quality,
        inputs=[
            ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity
        ],
        outputs=output
    )


# Launch app
app.launch(share=True)
