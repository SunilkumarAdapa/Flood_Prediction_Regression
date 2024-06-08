import os
import pandas as pd
import gradio as gr
import joblib
from src.source.model_training import ModelTrainerConfig

# Load the model
model_path = os.path.join(ModelTrainerConfig().model_dir, "RandomForest.joblib")
model = joblib.load(model_path)

# Prediction Function
def predict(MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization,
            ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments,
            IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides,
            Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss, InadequatePlanning,
            PoliticalFactors):

    input_data = pd.DataFrame([[
        MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization,
        ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments,
        IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides,
        Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss, InadequatePlanning,
        PoliticalFactors
    ]], columns=[
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation', 'Urbanization',
        'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices', 'Encroachments',
        'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability', 'Landslides',
        'Watersheds', 'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss', 'InadequatePlanning',
        'PoliticalFactors'
    ])
    
    predictions = model.predict(input_data)
    return predictions[0]

# Gradio Interface
inputs = [
    gr.Number(label='MonsoonIntensity'),
    gr.Number(label='TopographyDrainage'),
    gr.Number(label='RiverManagement'),
    gr.Number(label='Deforestation'),
    gr.Number(label='Urbanization'),
    gr.Number(label='ClimateChange'),
    gr.Number(label='DamsQuality'),
    gr.Number(label='Siltation'),
    gr.Number(label='AgriculturalPractices'),
    gr.Number(label='Encroachments'),
    gr.Number(label='IneffectiveDisasterPreparedness'),
    gr.Number(label='DrainageSystems'),
    gr.Number(label='CoastalVulnerability'),
    gr.Number(label='Landslides'),
    gr.Number(label='Watersheds'),
    gr.Number(label='DeterioratingInfrastructure'),
    gr.Number(label='PopulationScore'),
    gr.Number(label='WetlandLoss'),
    gr.Number(label='InadequatePlanning'),
    gr.Number(label='PoliticalFactors'),
]
outputs = gr.Textbox(label="Flood Probability")

interface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Flood Prediction Model")
interface.launch()
