from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import os
import logging
import time
import streamlit as st



class ModelTraining():
    def __init__(self, model_name="prophet_model"):
        self.model_name = model_name
        self.parameters = {
            'changepoint_prior_scale': 0.1,
            'changepoint_range': 0.2,
            'interval_width': 0.3,
            'uncertainty_samples': 100
        }

    def accuracy_measures(self, y_test,predictions,avg_method):
        """
        Calculates and logs regression metrics (MAE, MSE, R2 score).
        """
        # Regression Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log the metrics instead of printing
        logging.info("Regression Metrics")
        logging.info("--------------------")
        logging.info(f"MAE: {mae}")
        logging.info(f"MSE: {mse}")
        logging.info(f"R2 Score: {r2}")

        return mae, mse, r2
    

    def train_prophet_model(self, data_train, data_test, train_size, date, target):
        """
        Trains a Prophet model on the provided training data and evaluates its performance.
        """
        try:
            # Save train and test datasets to DVC
            train_file_path = "data/train.csv"

            # # Save the train and test df as CSV
            data_train.to_csv(train_file_path, index = False)

            # track train dataset with DVC
            #save_to_dvc([train_file_path])
            
            # Track dataset version with DVC (get the commit hash of the dataset)
            #dvc_version = dvc.api.get_url(train_file_path, rev="HEAD", remote="myremote")
            #logging.info(f"Using DVC version: {dvc_version}")


            #runname = "prophet-run-" + str(datetime.datetime.now()).replace(" ","T")
            with mlflow.start_run():

                # Initialize the Prophet model and data used for train and test
                prophet_model = Prophet(
                    changepoint_prior_scale=self.parameters['changepoint_prior_scale'],  # Allow for more rapid trend changes
                    changepoint_range=self.parameters['changepoint_range'],  # Use the first 90% of the data for changepoints
                    interval_width=self.parameters['interval_width'],  # 95% confidence intervals
                    uncertainty_samples=self.parameters['uncertainty_samples']   # Number of uncertainty simulations
                )

                # Add the S&P500 data as an additional regressor to improve predictions
                #prophet_model.add_regressor('Close_S&P500')

                # Fit the model on the merged data
                prophet_model.fit(data_train[[date, target]])

                # Forecast future prices
                forecast = prophet_model.predict(data_test[[date]])

                mae, mse, r2 = self.accuracy_measures(data_test[target],forecast["yhat"],'weighted')

                # Log parameters 
                self.log_model_parameters()

                # Log metrics
                self.log_metrics(mae, mse, r2)

                # Log the trained model
                mlflow.prophet.log_model(prophet_model, self.model_name)

        except Exception as e:
            logging.error(f"Error in training Prophet model: {e}")
            return None
        
    def retrain_load_prophet(self, data_train, data_test, train_size):
        """
        Retrains the Prophet model and loads the latest run.

        Parameters:
            data_train (DataFrame): The training data.
            data_test (DataFrame): The testing data.
            train_size (float): The train-test split ratio.

        Returns:
            Loaded Prophet model.
        """
        start_time = time.time()
        logging.info('start retraining ... ')

        try:
            # retrain prophet model
            self.train_prophet_model(data_train, data_test, train_size, 'ds', 'y')

            # Time taken for Prophet model training
            st.session_state.prophet_model_training_duration.observe(time.time() - start_time)
            logging.info('end retraining ... ')

            # Increment prophet model retraining counter
            st.session_state.prophet_model_retrain_count.inc() 

            # Retrieve the latest run
            runs = mlflow.search_runs(order_by=["start_time desc"])

            # Check if any runs were found
            if not runs.empty:
                logging.info("runs found ... ")
                # Get the most recent run
                latest_run = runs.iloc[0]  
                run_id = latest_run.run_id
                logging.info('last prophet model is fetched ... ')

                # Load the model from the most recent run
                model_uri = f"runs:/{run_id}/prophet_model"
                return mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            logging.error(f"Error in retraining Prophet model: {e}")
            st.session_state.app_error_count.inc()
            return None

    def log_model_parameters(self):
            # Log hyperparameters
            mlflow.log_param("changepoint_prior_scale", self.parameters['changepoint_prior_scale'])
            mlflow.log_param("changepoint_range", self.parameters['changepoint_range'])
            mlflow.log_param("interval_width", self.parameters['interval_width'])
            mlflow.log_param("uncertainty_samples", self.parameters['uncertainty_samples'])
            #mlflow.log_param("dvc_train_data_version", dvc_version)
            mlflow.log_param("model_file", self.model_name)

    def log_metrics(self, mae, mse, r2):
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
