import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        bid_opening: float,
        bid_highest: float,
        bid_lowest: float,
        bid_change: float,
        ask_open: float,
        ask_highest: float,
        ask_lowest: float,
        ask_close: float,
        ask_change: float):

        self.bid_opening = bid_opening

        self.bid_highest = bid_highest

        self.bid_lowest = bid_lowest

        self.bid_change = bid_change

        self.ask_open = ask_open

        self.ask_highest = ask_highest

        self.ask_lowest = ask_lowest

        self.ask_close = ask_close

        self.ask_change = ask_change

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "bid_opening": [self.bid_opening],
                "bid_highest": [self.bid_highest],
                "bid_lowest": [self.bid_lowest],
                "bid_change": [self.bid_change],
                "ask_open": [self.ask_open],
                "ask_highest": [self.ask_highest],
                "ask_lowest": [self.ask_lowest],
                "ask_close": [self.ask_close],
                "ask_change": [self.ask_change],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)