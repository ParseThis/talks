# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from time import time
import random

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the wine-quality csv file from the URL
    FILENAME = "winequality-red.csv"
    csv_url = \
    f"http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/{FILENAME}"

    if os.path.isfile(FILENAME):
        data = pd.read_csv(FILENAME, sep=";")
    else:
        try:
            data = pd.read_csv(csv_url, sep=";")
        except Exception as e:
            message = (f"Unable to download training & test CSV, check your\n"
            "internet connection. Error: {e}") 
            logger.exception(message)
        finally:
            sys.exit(1)


    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # TALK NOTE Linear regression with combined L1 and L2 priors as regularizer.
    # alpha is weighting both losses ( how regularization we want) 
    # 0 l1_ratio is mixing paramter, regularizing using ush L2 regularization
    # This"ll set both params  to 0.5 by default 

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # TALK NOTE couldn't get this to work well by just setting the env variable
    # remote_server_uri = os.environ.get("MLFLOW_TRACKING_URL", None) # set to your server URI
    # mlflow.set_tracking_uri(remote_server_uri)

    # TALK NOTE mlflow.start_run() return a ActiveRun with serves as a context manager
    # also create mlruns/ dir that support the ui (if the folder doesn"t exit)
    with mlflow.start_run(): # 
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        
        # TALK NOTE let time these predictions
        start_time = time()
        predicted_qualities = lr.predict(test_x)
        time_cost  = time() - start_time
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # TALK NOTE let make slightly interesting
        # by addding a random inference speed
        # which we calling 'time_cost'
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("time_cost",
                          random.randint(0, 10))

        # TALK NOTE we"re also serializing the elastic net model to disk
        mlflow.sklearn.log_model(sk_model=lr, 
                                 artifact_path="model",
                                 # registered_model_name="WineClassifier" 
                                )
