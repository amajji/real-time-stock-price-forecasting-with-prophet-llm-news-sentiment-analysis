from prophet import Prophet
import joblib



def prophet_model(data_train, data_test, date, target, retrain=True):
    if retrain:
        # Initialize the Prophet model
        prophet_model = Prophet(
            changepoint_prior_scale=0.1,  # Allow for more rapid trend changes
            changepoint_range=0.2,  # Use the first 90% of the data for changepoints
            interval_width=0.3,  # 95% confidence intervals
            uncertainty_samples=1000  # Number of uncertainty simulations
        )

        # Add the S&P500 data as an additional regressor to improve predictions
        #prophet_model.add_regressor('Close_S&P500')

        # Fit the model on the merged data
        prophet_model.fit(data_train[[date, target]])

        # Save the model with joblib
        joblib.dump(prophet_model, 'weights/prophet_model.joblib')

    # Load the model with joblib
    loaded_model = joblib.load('weights/prophet_model.joblib')

    # Predict future values
    #forecast = prophet_model.predict(data_test[['ds', 'Close_S&P500']])
    forecast = loaded_model.predict(data_test[[date]])

    # Extract the forecasted values and dates
    forecasted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return data_train, data_test, forecasted_values