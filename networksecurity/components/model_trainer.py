import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from xgboost import XGBClassifier

import mlflow
import dagshub
dagshub.init(repo_owner='4ashutosh98', repo_name='end-to-end-ml-project-mlflow-aws', mlflow=True)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def track_mlflow(self, best_model, classification_metric: ClassificationMetricArtifact):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            recall_score = classification_metric.recall_score
            precision_score = classification_metric.precision_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            mlflow.sklearn.log_model(best_model, "model")




    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "LogisticRegression": LogisticRegression(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBoostClassifier": XGBClassifier()
        }

        params = {
            "LogisticRegression": {
                "C": [0.1, 1, 10]
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            },
            "DecisionTreeClassifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": [None, 5, 10, 15, 20, 25, 30],
                "max_features": ["sqrt", "log2"],
                "min_samples_split": [2, 5, 10]
            },
            "AdaBoostClassifier": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "learning_rate": [0.001, 0.01, 0.05, 0.1]
            },
            "GradientBoostingClassifier": {
                "loss": ["log_loss", "exponential"],
                "learning_rate": [0.001, 0.01, 0.05, 0.1],
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "criterion": ["friedman_mse", "squared_error"],
                "max_features": ["sqrt", "log2"],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
            },
            "RandomForestClassifier": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_features": ["sqrt", "log2"],
                "max_depth": [None, 5, 10, 15, 20, 25, 30]
            },
            "XGBoostClassifier": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "learning_rate": [0.001, 0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
            }
        }

        model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        print(f"Model report: {model_report} \n ==============================")

        ## Get the best model score from the report dict
        best_model_score = max(sorted(model_report.values()))

        ## Get the best model name from the report dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        
        best_model = models[best_model_name]
        print(f"Best model: {best_model} \n ==============================")
        print(f"Best model score: {best_model_score} \n ==============================")

        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_train, y_train_pred)

        ## Track the model training using MLFLOW
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_test, y_test_pred)

        ## Track the model testing using MLFLOW
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)    
        model_dir_path = os.path.dirname(self.model_trainer_config.model_trained_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(
            model=best_model,
            preprocessor=preprocessor
        )

        save_object(
            self.model_trainer_config.model_trained_file_path,
            obj = NetworkModel
        )

        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.model_trained_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## Loading training numpy array and testing numpy array
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            ## Splitting the data into features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model_trainer_artifact = self.train_model(
                X_train, y_train, X_test, y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e