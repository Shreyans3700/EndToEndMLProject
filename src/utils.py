from src.logger import logging
from src.exception import CustomException
import os
import sys
import dill
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_object(object, file_path: str) -> bool:
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj=object, file=file_obj)

        return True

    except Exception as e:
        raise CustomException(e, sys)


def calculate_model_score(y, y_pred, prefix):
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2_scr = r2_score(y, y_pred)

    return {
        f"{prefix} Mean Absolute Error": mae,
        f"{prefix} Mean Squared Error": mse,
        f"{prefix} Root Mean Squared Error": rmse,
        f"{prefix} R2 Score": r2_scr,
    }


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models.values()))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            logging.info("Starting Model Training of %s", model_name)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            training_data_model_report = calculate_model_score(
                y_train, y_train_pred, "Training"
            )
            testing_data_model_report = calculate_model_score(
                y_test, y_test_pred, "Testing"
            )

            model_report = {
                "Training Data Report": training_data_model_report,
                "Test Data Report": testing_data_model_report,
            }

            report[list(models.keys())[i]] = model_report

        return report

    except Exception as e:
        raise CustomException(e, sys)
