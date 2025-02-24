################################################################################################################

#                                             Author: Anass MAJJI                                              #
 
#                                             File Name: app.py                                                #

#                                       Creation Date: January 01, 2025                                        #

#                                          Source Language: Python                                             #

#  Repository:  https://github.com/amajji/real-time-crypto-forecasting-with-arima-prophet-llm-news-sentiment    #

#                                          --- Code Description ---                                            #

#    Deploy LLM RAG Chatbot with Langchain on a Streamlit web application using only CPU                       #

################################################################################################################


################################################################################################################
#                                                  Packages                                                    #
################################################################################################################
import sys
import os
import streamlit as st
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import mlflow
from datetime import datetime
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.prometheus_metrics import PrometheusMetrics
from src.sentiment_analysis import SentimentAnalysis
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from src.visualization import Visualization
from src.investment_tracker import InvestmentTracker
from src.lstm_arima_models import LstmArimaModels
################################################################################################################
#                                                   Main code                                                  #
################################################################################################################

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the URI for the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#mlflow.set_tracking_uri("http://mlflow:5000")

# Set the experiment name for MLflow 
mlflow.set_experiment("mlflow-prophet-model")


# Main streamlit page
def main():
    # Set layout as wide
    st.set_page_config(layout="wide")
    
    # Define the title
    st.title("✨ Stock Price forecasting Dashboard")

    # get current date
    current_date = datetime.now()

    # date feature
    date = "Date"

    # Initialize Prometheus metrics
    prometheus_metrics_inc = PrometheusMetrics()

    # Create an instance of SentimentAnalysis
    sentiment_analyzer_inc = SentimentAnalysis()

    # Create an instance of LstmArimaModels
    lstm_arima_models_inc = LstmArimaModels()

    # Create an instance of ModelTraining
    model_training_inc = ModelTraining()

    # Create an instance of Visualization
    visualization_inc = Visualization()

    # Instantiate the DataPreprocessing class
    data_preprocessor_inc = DataPreprocessing(current_date=current_date, sentiment_analyzer=sentiment_analyzer_inc)

    # Instanciate the investment tracker class
    investment_tracker_inc = InvestmentTracker(date=date)

    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer =  sentiment_analyzer_inc.load_roberta_model()

    if 'prophet' not in st.session_state:
            st.session_state.prophet_model = None

    if 'merged_data' not in st.session_state:        
        # Load data (both price and news)
        data_preprocessor_inc.load_data()

        # Merge data (BTC prices with news data)
        data_preprocessor_inc.merge_data()

        # Process the data
        data_preprocessor_inc.processing_data()

        # save merged_data 
        st.session_state.merged_data = data_preprocessor_inc.df_merged_data

    # Define x_min, x_max, y_min and y_max
    x_min, x_max, y_min, y_max = st.session_state.merged_data['ds'].min(), st.session_state.merged_data['ds'].max(), st.session_state.merged_data['y'].min(), st.session_state.merged_data['y'].max()

    # Define 5 columns 
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.subheader("Price forcasting model") 
        # Create 3 columns for price forecasting models
        col6, col7, col8 = st.columns(3)
        with col6:
            # checkbox for Arima model
            model_arima = st.checkbox("ARIMA")

        with col7:
            # checkbox for LSTM model
            model_lstm = st.checkbox("LSTM")

        with col8:
            # checkbox for Prophet model
            model_prophet = st.checkbox("Prophet")

    with col2:
        st.subheader("News analysis model") 
        # Create 2 columns for news analysis models
        col9, col10 = st.columns(2)
        with col9:
            # Checkbox for DeepSeek model
            model_bert = st.checkbox("LLM")

        with col10:
            # Checkbox for DistilBert model
            model_llama = st.checkbox("DistilBert")

    with col3:
        # Create a container for dynamic plot updates
        portfolio_placeholder = st.empty()

    with col4:
        # Create a container to show the current price
        price_placeholder = st.empty()

        # Create a container for dynamic plot updates
        plot_model_perf = st.empty()

    with col5:
        # Create a container to show the error of the model
        error_placeholder = st.empty()

    # define 2 columns 
    col11, col12 = st.columns(2)

    with col11:
        # Create a container for dynamic plot updates
        plot_placeholder = st.empty()

        # define 2 columns 
        col13, col14 = st.columns(2)
        with col13:
            # Create date input widgets for start date selection
            start_date = st.date_input(
                'Select start date', 
                min_value=st.session_state.merged_data['ds'].min(), 
                max_value=st.session_state.merged_data['ds'].max(), 
                value=st.session_state.merged_data['ds'].min())
        
        with col14:
            # Create date input widgets for end date selection
            end_date = st.date_input(
                'Select end date', 
                min_value=st.session_state.merged_data['ds'].min(), 
                max_value=st.session_state.merged_data['ds'].max(), 
                value=st.session_state.merged_data['ds'].max())

    with col12:
        # Create a container for models performance
        plot_perf_model = st.empty()

        # define col 5 an 6
        col15, col16 = st.columns(2)
        with col15:
            # Input for amount to invest
            st.session_state.new_invested_amount = st.number_input('Enter the amount you want to invest:',min_value=0, step=1)
            st.session_state.investment_amount += st.session_state.new_invested_amount 
            
        with col16:
            # CSS code for Buy button
            st.markdown(visualization_inc.plot_buy_button(), unsafe_allow_html=True)

            # Session state to store whether the button was pressed
            if 'invested' not in st.session_state:
                st.session_state.invested = False

            # Button to calculate P&L
            if st.button('Buy'):
                st.session_state.invested = True

    # Filter on dates between start_date and end_date
    st.session_state.merged_data = st.session_state.merged_data.query(f"@start_date<=ds and ds<=@end_date")

    # Initialize PNL and ROI with 0
    pnl, roi = 0, 0 

    # Initialize boolean flags
    bool_press, bool_first_time = False, True

    # Initialize colors features
    color_price, portfolio_color = 'gray', 'darkblue'

    # Lists of true and predicted values and dates
    y_pred, y_true, dates = [], [], []

    # number of days to consider to assess the MAE of the model
    nb_days = 5

    compt = 0 

    # If one of the button is pressed
    if model_arima or model_lstm or model_prophet:
        # Spinner for long operations
        with st.spinner('Processing...'):
            for train_size in range(50, len(st.session_state.merged_data), 150):
                # Define train and test datasets
                data_train, data_test = st.session_state.merged_data[:train_size], st.session_state.merged_data[train_size:] 
                
                # If ARIMA was selected
                if model_arima:
                    # Get forecasted values using ARIMA model
                    data_train, data_test, forecasted_values = lstm_arima_models_inc.arima_model(data_train, data_test, 'ds', 'y', retrain=False)

                # if LSTM was selected
                if model_lstm:
                    # Get forecasted values using LSTM model
                    data_train, data_test, forecasted_values = lstm_arima_models_inc.lstm_model(data_train, data_test, 'ds', 'y', retrain=False)

                # if PROPHET was selected
                if model_prophet:

                    # Retrieve the latest run
                    runs = mlflow.search_runs(order_by=["start_time desc"])

                    # train the model for the first time 
                    if not st.session_state.prophet_model: 
                        st.session_state.prophet_model = model_training_inc.retrain_load_prophet(data_train, data_test, train_size)

                    # start time
                    start_time = time.time()

                    # Predict future values
                    forecast = st.session_state.prophet_model.predict(data_test[['ds']])
                    
                    # Track request duration
                    st.session_state.request_duration.observe(time.time() - start_time)

                    # Extract the forecasted values and dates
                    forecasted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                    print("Shape of predictions : ", st.session_state.df_predictions.shape)
                    # retrain the model when the MAE is greater than 1000$ during the last nb_days
                    if len(st.session_state.df_predictions)>0:
                        print("YESSSSSSSSS")
                        if compt < 4:
                            compt += 1
                            if mean_squared_error(st.session_state.df_predictions["y_true"][-nb_days:],  st.session_state.df_predictions["y_pred"][-nb_days:]) > 1000:
                                st.session_state.prophet_model = model_training_inc.retrain_load_prophet(data_train, data_test, train_size)
                                st.session_state.model_error_rate.set(0)
                            else:
                                st.session_state.model_error_rate.set(1)

                # current price 
                current_price = data_test['y'].iloc[0]

                # If not first iteration
                if not bool_first_time: 
                    # Append predicted value to y_pred
                    y_pred.append(next_predicted_value)

                    # Append current price to y_true
                    y_true.append(data_train['y'].iloc[-1])

                    # Add current date
                    dates.append(data_train['ds'].iloc[-1])

                    # Calculate MAE
                    model_error = mean_absolute_error(data_test['y'], forecasted_values['yhat'])

                    # Show model's error in colored cell
                    error_placeholder.markdown(visualization_inc.plot_model_mae(model_error), unsafe_allow_html=True)

                    # Define the color depending on the variation between old and new stock price
                    if current_price >= old_value:
                        color_price = 'green'
                    else:
                        color_price = 'red'
                    # Det the new price as old value
                    old_value = current_price

                # predicted value
                next_predicted_value = forecasted_values['yhat'].iloc[0]

                if bool_first_time:
                    old_value = current_price

                bool_first_time = False

                # Update values when Buy button is pressed
                if st.session_state.invested:
                    bool_press = True
                    investment_tracker_inc.update_investment(data_train)
                    
                # Compte PNL and ROI
                if bool_press: 
                    pnl = (current_price - st.session_state.new_stock_price)*st.session_state.nb_units
                    roi = 100*pnl/st.session_state.investment_amount

                # Define the colors depending on the sign of the pnl
                if pnl == 0:
                    color_pnl = 'gray'
                elif pnl > 0:
                    color_pnl = 'green'
                else:
                    color_pnl = 'red'

                # Show Current price in colored cell
                price_placeholder.markdown(visualization_inc.plot_current_price(color_price, current_price), unsafe_allow_html=True)

                # Show Portfolio details in colored cell
                portfolio_placeholder.markdown(visualization_inc.plot_portfolio_details(portfolio_color), unsafe_allow_html=True)

                # Show P&L in colored cell
                plot_model_perf.markdown(visualization_inc.plot_pnl(pnl, roi), unsafe_allow_html=True)

                # Figure to plot train, test and forecasted values 
                fig = visualization_inc.plot_forecast(data_train, data_test, forecasted_values)

                # if PROPHET was selected
                if model_prophet:
                    # Add uncertainty (fill between yhat_lower and yhat_upper)
                    fig.add_trace(go.Scatter(
                        x=forecasted_values['ds'],
                        y=forecasted_values['yhat_upper'],
                        mode='lines',
                        line={'color': 'rgba(255, 0, 0, 0.2)', 'width': 0},  # Make it invisible but fillable
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=forecasted_values['ds'],
                        y=forecasted_values['yhat_lower'],
                        mode='lines',
                        line={'color': 'rgba(255, 0, 0, 0.2)', 'width': 0},  # Make it invisible but fillable
                        fill='tonexty',  # Fill between this trace and the previous one
                        fillcolor='rgba(255, 0, 0, 0.2)',  # Set the fill color
                        showlegend=False
                    ))

                # Layout customization
                fig.update_layout(
                    title='Train, Test, and Forecasted Data',
                    xaxis_title='Date',
                    yaxis_title='Values',
                    xaxis=dict(range=[x_min, x_max]),  
                    yaxis=dict(range=[y_min, 1.5*y_max]), 
                    template="plotly",
                )
                # Update the plot in Streamlit
                plot_placeholder.plotly_chart(fig)

                # Figure to plot true and predicted values
                fig_1 = visualization_inc.plot_true_forecasted_values(dates, y_true, y_pred, x_min, x_max)

                # Update the plot in Streamlit
                plot_perf_model.plotly_chart(fig_1)

                # save date, true value and predicted value
                st.session_state.df_predictions.loc[len(st.session_state.df_predictions)] = [data_train['ds'].iloc[-1], data_train['y'].iloc[-1], next_predicted_value]
            # st.session_state.df_predictions["Date"] = dates
            # st.session_state.df_predictions["y_true"] = y_true
            # st.session_state.df_predictions["y_pred"] = y_pred

        st.session_state.df_predictions.to_excel("df_predictions.xlsx")



if __name__ == "__main__":
    main()