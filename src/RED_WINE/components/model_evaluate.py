import pandas as pd
from RED_WINE.utils.utils import save_json
from RED_WINE.logging.logger import logger
from RED_WINE.entity.config_entity import ModelEvaluateConfig
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
import numpy as np
import joblib
from urllib.parse import urlparse
from pathlib import Path


class ModelEvaluate:
    def __init__(self , config= ModelEvaluateConfig):
        self.config = config

    def eval_metrics(self,actual , pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    def start_model_evaluation(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        x_test = test_data.drop(columns=self.config.target_column_name,axis=1)
        y_test = test_data[self.config.target_column_name]

        y_pred = model.predict(x_test)

        rmse , mae ,r2 = self.eval_metrics(y_test,y_pred )

        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)