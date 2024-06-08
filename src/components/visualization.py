import os
import pandas as pd
from ydata_profiling import ProfileReport
from dataclasses import dataclass
import sys
from src.exception import CustomException
from src.logger import logging


@dataclass
class VisualizationConfig:
    profile_report_path: str = os.path.join('artifacts', "EDA_report.html")

class DataVisualization:
    def __init__(self):
        self.config = VisualizationConfig()

    def generate_profile_report(self, data_path):
        logging.info("Entered the data visualization method or component")
        try:
            df = pd.read_csv(data_path)
            logging.info("Read the data as dataframe")

            # Generate the profiling report
            profile = ProfileReport(df, title="EDA Profiling Report", explorative=True)
            logging.info("Generated the profile report")

            # Save the report to the specified path
            os.makedirs(os.path.dirname(self.config.profile_report_path), exist_ok=True)
            profile.to_file(self.config.profile_report_path)
            logging.info(f"Saved the profile report to {self.config.profile_report_path}")

            return self.config.profile_report_path
        except Exception as e:
            logging.error("Error during data visualization")
            raise CustomException(e, sys)

                
            
        
        

            
