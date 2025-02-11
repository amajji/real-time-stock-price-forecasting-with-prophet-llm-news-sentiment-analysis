from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import datetime
 

def accuracy_measures(y_test,predictions,avg_method):
    # Regression Metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Regression Metrics")
    print("--------------------")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    return mae, mse, r2



def train_prophet_model(data_train, data_test, date, target):
    runname = "prophet-run-" + str(datetime.datetime.now()).replace(" ","T")
    with mlflow.start_run(run_name=runname) as prophet_run:

        changepoint_prior_scale=0.1
        changepoint_range=0.2
        interval_width=0.3
        uncertainty_samples=1000

        # Initialize the Prophet model
        prophet_model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,  # Allow for more rapid trend changes
            changepoint_range=changepoint_range,  # Use the first 90% of the data for changepoints
            interval_width=interval_width,  # 95% confidence intervals
            uncertainty_samples=uncertainty_samples  # Number of uncertainty simulations
        )

        # Add the S&P500 data as an additional regressor to improve predictions
        #prophet_model.add_regressor('Close_S&P500')

        # Fit the model on the merged data
        prophet_model.fit(data_train[[date, target]])

        # Forecast future prices
        forecast = prophet_model.predict(data_test[[date]])

        mae, mse, r2 = accuracy_measures(data_test[target],forecast["yhat"],'weighted')


        # Log hyperparameters
        mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
        mlflow.log_param("changepoint_range", changepoint_range)
        mlflow.log_param("interval_width", interval_width)
        mlflow.log_param("uncertainty_samples", uncertainty_samples)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log the trained model
        mlflow.prophet.log_model(prophet_model, "prophet_model")




