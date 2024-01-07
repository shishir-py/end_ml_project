import os
import sys
import pandas as pd
import numpy as np
# from src.Exception import custom_exception
from pickle4 import pickle
from src.Exception import custom_exception


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            # print("done")
    except Exception as e:
        # print(f"Error: {e}")
        return custom_exception(e, sys)

file_path = "artifacts/"
obj = 1
save_obj(file_path, obj)
