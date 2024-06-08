import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.source.data_transformation import DataTransformation
from src.source.data_transformation import DataTransformationConfig

    
from src.source.visualization import DataVisualization
from src.source.visualization import VisualizationConfig

from src.source.model_training import ModelTrainer
from src.source.model_training import ModelTrainerConfig

from src.source.evaluate import ModelEvaluator
from src.source.evaluate import EvaluationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'data\Data\train.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    
    visualization=DataVisualization()
    profile_report_path = visualization.generate_profile_report(train_data)
    
    model_trainer = ModelTrainer()
   # train_data_path = 'artifacts/Transformed_train.csv'
   # test_data_path = 'artifacts/Transformed_test.csv'
    results = model_trainer.train_and_evaluate()
    
    evaluator = ModelEvaluator()
    results = evaluator.generate_evaluation_report()
    
    #logging.info(f"EvaluationResults: {evaluation_report}")
    

    
        

       
    
    
    

    