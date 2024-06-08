import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_train_data_path=os.path.join('artifacts',"Transformed_train.csv")
    preprocessor_test_data_path = os.path.join('artifacts',"Transformed_test.csv")
    preprocessor_obj_file_path=os.path.join('artifacts',"Transformed.pkl")
#DataTransformation     
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):
        logging.info("Entered the data transformation")
        
        try:
            train_df = pd.read_csv(train_path).drop(["id"], axis=1)
            test_df = pd.read_csv(test_path).drop(["id"], axis=1)

            logging.info("read the train and test dataframe")
            
            #StandardScaler
            scaler = StandardScaler()
            
            #the data frame conitains only numerical features only
            train_df_scaled = scaler.fit_transform(train_df)
            test_df_scaled = scaler.fit_transform(test_df)
            
            #convert back to the data frame:
            train_df_scaled = pd.DataFrame(train_df_scaled,columns=train_df.columns)
            test_df_scaled = pd.DataFrame(test_df_scaled,columns=test_df.columns)
            
            #save transform data
            train_df_scaled.to_csv(self.data_transformation_config.preprocessor_train_data_path,index=False,header=True)
            test_df_scaled.to_csv(self.data_transformation_config.preprocessor_test_data_path,index=False,header=True)
            logging.info("Saved the transformation train and test data")
            
            #save the transformer object:
            joblib.dump(scaler,self.data_transformation_config.preprocessor_obj_file_path)
            logging.info("saved the scaler object")
            
            return(
                self.data_transformation_config.preprocessor_train_data_path,
                self.data_transformation_config.preprocessor_test_data_path,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
            
        except Exception as e:
            logging.error("Error During the data transformation")
            raise CustomException(e,sys)
            
        