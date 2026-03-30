from src.logger import logging
from src.exception import CustomException
import os
import sys
import dill
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV


def save_object(object, file_path: str) -> bool:
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj=object, file=file_obj)

        return True

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file=file_path, mode="rb") as file_obj:
            return dill.load(file_obj)
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


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        trained_models = {}  # 🔥 ADD THIS

        for i in range(len(list(models.values()))):
            model = list(models.values())[i]
            param = list(params.values())[i]
            model_name = list(models.keys())[i]

            logging.info("Starting Model Training of %s", model_name)

            random_cv = RandomizedSearchCV(
                estimator=model,
                cv=3,
                param_distributions=param,
                n_jobs=-1,
                n_iter=50,
                random_state=42,
            )

            random_cv.fit(X_train, y_train)

            # 🔥 STORE TRAINED GRID
            trained_models[model_name] = random_cv

            y_train_pred = random_cv.predict(X_train)
            y_test_pred = random_cv.predict(X_test)

            training_data_model_report = calculate_model_score(
                y_train, y_train_pred, "Training"
            )
            testing_data_model_report = calculate_model_score(
                y_test, y_test_pred, "Testing"
            )

            model_report = {
                "Training Data Report": training_data_model_report,
                "Test Data Report": testing_data_model_report,
                "Best Params": random_cv.best_params_,
            }

            report[model_name] = model_report

        return report, trained_models  # 🔥 RETURN BOTH

    except Exception as e:
        raise CustomException(e, sys)
