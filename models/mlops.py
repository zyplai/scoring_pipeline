import os
from datetime import datetime

import pandas as pd
import mlflow
import catboost as cb

from configs.config import settings


def mlflow_track(
    model: cb.CatBoostClassifier,
    df: pd.DataFrame,
    auc_train: float,
    auc_test: float,
    run_time: datetime
) -> None:

    mlflow.set_tracking_uri("http://13.81.254.179:5050")
    if not mlflow.get_experiment_by_name(settings.MLFLOW.experiment_name):
        mlflow.create_experiment(settings.MLFLOW.experiment_name)
    mlflow.set_experiment(settings.MLFLOW.experiment_name)
    with mlflow.start_run(run_name=f'run_{run_time}'):
        # Log parameters, metrics, and artifacts
        mlflow.log_params(settings.SET_FEATURES.model_params)
        mlflow.log_metric("train_auc", auc_train)
        mlflow.log_metric("test_auc", auc_test)

        # Save the model as an MLflow artifact
        model_path = "model"
        from mlflow.models import infer_signature
        signature = infer_signature(df[df['is_train'] == 0][settings.SET_FEATURES.features_list], df['predictions'].values)
        mlflow.catboost.log_model(model, model_path, signature=signature)
        # Set the artifact path for the current run
        os.path.join(mlflow.get_artifact_uri(), model_path)
