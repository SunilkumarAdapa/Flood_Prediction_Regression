import os
import sys
import joblib
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join('artifacts', "models")
    train_data_path: str = os.path.join('artifacts', 'Transformed_train.csv')
    test_data_path: str = os.path.join('artifacts', 'Transformed_test.csv')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(self.config.model_dir, exist_ok=True)

    def save_model(self, model, model_name):
        try:
            model_path = os.path.join(self.config.model_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            logging.info(f"Saved {model_name} model to {model_path}")
            return model_path
        except Exception as e:
            logging.error(f"Error saving the model {model_name}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return mse, mae, r2
        except Exception as e:
            logging.error("Error evaluating the model")
            raise CustomException(e, sys)

    def train_and_evaluate(self):
        try:
            # Load the data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            # Separate features and target
            X_train = train_df.drop(columns="FloodProbability")
            y_train = train_df["FloodProbability"]
            X_test = test_df.drop(columns="FloodProbability")
            y_test = test_df["FloodProbability"]

            # Models to train
            models = {
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'XGBoost': xgb.XGBRegressor(),
            }

            results = {}
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                mse, mae, r2 = self.evaluate_model(model, X_test, y_test)
                results[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2_Score': r2
                }
                logging.info(f"{model_name} - MSE: {mse}, MAE: {mae}, R2_Score: {r2}")
                self.save_model(model, model_name)

            return results

        except Exception as e:
            logging.error("Error during model training and evaluation")
            raise CustomException(e, sys)
