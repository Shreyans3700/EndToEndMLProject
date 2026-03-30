import sys
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import evaluate_models
from src.utils import save_object
from src.configs.models_hyperparameter_configs import ModelsHyperParameterConfigs


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],  ## All rows and all cols till last one
                train_arr[:, -1],  ## pick the last column from all rows
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Support Vector Regressor": SVR(),
                "KN Regressor": KNeighborsRegressor(),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "XG Boost": XGBRegressor(),
            }

            model_params = ModelsHyperParameterConfigs()

            model_report, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, model_params.params
            )

            logging.info("Model Reports are %s", model_report)

            test_r2_scores = [
                scr["Test Data Report"]["Testing R2 Score"]
                for scr in model_report.values()
            ]

            best_model_score_index = test_r2_scores.index(max(test_r2_scores))

            best_model_name = list(model_report.keys())[best_model_score_index]

            best_model_grid = trained_models[best_model_name]

            save_object(
                best_model_grid, self.model_trainer_config.trained_model_file_path
            )

            best_score = test_r2_scores[best_model_score_index]

            if best_score < 0.6:
                logging.info("No best model found in both training and testing.")
                raise CustomException("No Best Model Found")

            logging.info(
                f"Best model found: {best_model_name} with R2 score: {best_score}"
            )

        except Exception as e:
            raise CustomException(e, sys)
