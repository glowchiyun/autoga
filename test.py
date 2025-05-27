import random
import pandas as pd
import numpy as np
import load_data as ld
import  normalizer as nl 
import imputer as imp
import outlier_detector as out
import duplicate_detector as dup
import feature_selector as fs
import classifier as cl
import clusterer as ct
import time
import regressor as rg
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    data = ld.load_data("datasets/Bank_churn_modelling.csv")
    target = "Exited"
    data = imp.Imputer(data.copy(), strategy='EM').transform()
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target,axis=1),
        data[target],
        test_size=0.25,
        random_state=42
    )
    df= {"train": X_train,
         "test": X_test,
         "target": y_train,
         "target_test": y_test}
    score = cl.Classifier(
        dataset=df,
        target = target,
        strategy='NB'
        ).transform()
    print(score)