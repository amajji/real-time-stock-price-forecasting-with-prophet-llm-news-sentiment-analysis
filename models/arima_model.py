from pmdarima import auto_arima
import joblib
import pandas as pd


def arima_model(data_train, data_test, date, target, retrain=False):

    if retrain:
        # Fit the model with auto_arima
        model = auto_arima(data_train[[target]], 
                        seasonal=False,  # Set to True if your data has a seasonal component
                        m=1,             # Seasonality period, 1 means no seasonality
                        stepwise=True,   # Uses stepwise search for optimal parameters
                        trace=True)      # Print the search process

        # Save the model
        joblib.dump(model, 'weights/arima_model.pkl')

    # Load the saved model
    loaded_model = joblib.load('weights/arima_model.pkl')

    # Forecast the next 10 time points (adjust number of steps as needed)
    forecast_steps = len(data_test)
    forecasted_values = loaded_model.predict(n_periods=forecast_steps)

    # set test index to forcasted data

    # set test index to forcasted data
    forecasted_values = pd.DataFrame(forecasted_values, columns = ['yhat'])
    forecasted_values[date] = data_test[date]

    return data_train, data_test, forecasted_values