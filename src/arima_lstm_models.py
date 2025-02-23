from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
import joblib
import numpy as np
import pandas as pd



n_steps_ahead_value = 1

# Function to create sequences using both Bitcoin and S&P 500 data
def create_sequences_multivariate(data_in, time_step=6, n_steps_ahead=n_steps_ahead_value):
    """ 
    This function reshapes data_in to be 3D as expected by LSTM (samples, time steps, features)
    """
    X_out = []
    y_out = []
    for i in range(time_step, len(data_in)-n_steps_ahead + 1):
        X_out.append(data_in[i-time_step:i, :])  # Take the previous `time_step` data as input (both Bitcoin and S&P 500)
        y_out.append(data_in[i:i+n_steps_ahead, 0])  # Predict the next n_steps_ahead Bitcoin price
    return np.array(X_out), np.array(y_out)

def lstm_model(data_train, data_test, date, target, retrain=False):
    

    if retrain:
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale both Bitcoin and S&P 500 data (since we now have two features)
        data_train_norm = scaler.fit_transform(data_train[[target]])

        # Save the fitted scaler 
        joblib.dump(scaler, 'weights/minmax_scaler.joblib')

    # Load the saved scaler
    scaler_loaded = joblib.load('weights/minmax_scaler.joblib')

    # Scale both Bitcoin and S&P 500 data for test dataset
    data_train_norm = scaler_loaded.transform(data_train[[target]])

    # Scale both Bitcoin and S&P 500 data for test dataset
    data_test_norm = scaler_loaded.transform(data_test[[target]])

    # Create sequences with the train data (including both Bitcoin and S&P 500 features)
    X_train, y_train = create_sequences_multivariate(data_train_norm, n_steps_ahead=n_steps_ahead_value)

    # Create sequences with the test data (including both Bitcoin and S&P 500 features)
    X_test, y_test = create_sequences_multivariate(data_test_norm, n_steps_ahead=n_steps_ahead_value)

    if retrain:
        # Define the LSTM model
        model = Sequential()

        # Add LSTM layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50, return_sequences=False))

        # Add Dropout to prevent overfitting
        model.add(Dropout(0.2))

        # Add a Dense output layer
        model.add(Dense(units=n_steps_ahead_value))  # Predict the next Bitcoin price

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

        # Save the entire model (architecture + weights)
        model.save('weights/lstm_model.h5')  # Save the model to a file


    # Load the saved model
    loaded_model = load_model('weights/lstm_model.h5')

    # Make predictions on the test data
    forecasted_values = loaded_model.predict(X_test)

    # Inverse transform the predictions and actual values to get the actual Bitcoin price scale
    forecasted_values = scaler_loaded.inverse_transform(np.concatenate((forecasted_values, np.zeros((forecasted_values.shape[0], 1))), axis=1))[:,0]
    y_test_actual = scaler_loaded.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))), axis=1))[:,0]

    forecasted_values = pd.DataFrame(forecasted_values, columns = ['yhat'])
    forecasted_values[date] = list(data_test.iloc[-len(forecasted_values):][date])

    data_test_edit = pd.DataFrame(y_test_actual, columns = ['y'])
    data_test_edit[date] = list(data_test.iloc[-len(forecasted_values):][date])

    return data_train, data_test_edit, forecasted_values


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