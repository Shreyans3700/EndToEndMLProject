from src.logger import logging
from src.exception import CustomException
import os
import sys
import dill


def save_object(object, file_path: str) -> bool:
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj=object, file=file_obj)

        return True

    except Exception as e:
        raise CustomException(e, sys)
