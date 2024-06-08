import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
from dataclasses import dataclass
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class EvaluationConfig:
    model_dir: str =os.path.join('artifacts','models')
    test_data_path: str = os.path.join('artifacts','Transformed_test.csv')
    evaluation_report_path: str = os.path.join('artifacts','evaluation_report.txt')
    
class ModelEvaluator:
    def __init__(self):
        self.config =EvaluationConfig()
        
    def load_model(self,model_name):
        try:
            model_path = os.path.join(self.config.model_dir,f"{model_name}.joblib")
            model = joblib.load(model_path)
            logging.info(f"Loaded {model_name} model from {model_path}")
            
            return model
        except Exception as e:
            logging.error(f"Error loading {model_name} model")
            raise CustomException(e,sys)
        
    def evaluate_model(self,model,X_test,y_test):
        predictions =model.predict(X_test)
        mse = mean_squared_error(y_test,predictions)
        mae = mean_absolute_error(y_test,predictions)
        r2 = r2_score(y_test, predictions)
        
        return mse, mae, r2
    
    def generate_evaluation_report(self):
        
        try:
            #load the data 
            test_df = pd.read_csv(self.config.test_data_path)
            logging.info("Read the test data frame")
            
            #Predictions and Target
            X_test = test_df.drop(columns='FloodProbability')
            y_test = test_df["FloodProbability"]
            logging.info("Prediction and Target")
            
            models = ['DecisionTree','RandomForest','GradientBoosting','XGBoost']
            results ={}
            
            for model_name in models:
                model =self.load_model(model_name)
                mse,mae,r2 = self.evaluate_model(model,X_test, y_test)
                results[model_name] = {
                    'MSE' : mse,
                    'MAE' : mae,
                    'R2_Score':r2
                }
                logging.info(f"{model_name} - MSE: {mse}, MAE: {mae}, R2_Score: {r2}")
                
                # Save evaluation results to a file
            with open(self.config.evaluation_report_path, 'w') as f:
                for model_name, metrics in results.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  MSE: {metrics['MSE']}\n")
                    f.write(f"  MAE: {metrics['MAE']}\n")
                    f.write(f"  R2 Score: {metrics['R2_Score']}\n")
                    f.write("\n")
            logging.info(f"Saved evaluation report to {self.config.evaluation_report_path}")

            return results
        
        except Exception as e:
            logging.error("Error during model evaluation")
            raise CustomException(e, sys)
                
            
            
            
            



