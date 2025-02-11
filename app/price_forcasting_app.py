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
import yfinance as yf
import praw
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.arima_model import arima_model
from models.lstm_model import lstm_model
from models.prophet_model import train_prophet_model
import mlflow
import torch
from datetime import datetime
import joblib

################################################################################################################
#                                                   Main code                                                  #
################################################################################################################


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("prophet-model")

def get_historical_news(current_date):
    # Initialize Reddit API client with your credentials
    reddit = praw.Reddit(
        client_id='rVofejEZJ37RP4Ht0Di9dg',  
        client_secret='KoytCCII7wPYlSY0Hf7jUpZ_yPn59g',  
        user_agent='BTC News Scraper v1'  
    )

    # Define the date range (2020 to now)
    start_date = datetime(2019, 1, 1)

    # data news
    data = []

    # Subreddit for Bitcoin news
    subreddit = reddit.subreddit('Bitcoin')

    # Fetch posts in chunks (Reddit limits you to top, hot, or new posts)
    for submission in subreddit.top(time_filter='all', limit=100000):  # Increase the limit as needed, based on rate limits
        post_date = datetime.utcfromtimestamp(submission.created_utc)
        if start_date <= post_date <= current_date:
            data.append(
                {"Title":submission.title,
                "URL":submission.url,
                #"Score":submission.score,
                "Published":post_date,
                        })
            
    # convert data to dataframe
    df_news = pd.DataFrame(data)

    # get only year-month-day format
    df_news["Published"] = df_news["Published"].apply(lambda x: str(x).split(" ")[0].strip())

    # set Published column as index
    df_news.set_index('Published', inplace=True, drop=True)

    return df_news



def get_historical_prices(current_date):
    # Fetch cryptocurrency data (e.g., Bitcoin)
    btc_data = yf.download('BTC-USD', start='2019-01-01', end=current_date)

    # Assume you have another dataset, such as S&P 500
    sp500_data = yf.download('^GSPC', start='2019-01-01', end=current_date)


    # Get only level 0 columns
    btc_data.columns = btc_data.columns.levels[0]
    sp500_data.columns = sp500_data.columns.levels[0]

    # merge both btc and sp500 dataframes
    btc_sp500_data  = pd.merge(btc_data, sp500_data, left_index = True, right_index = True, suffixes=('_BTC', '_S&P500'))

    # convert datetime format to string
    btc_sp500_data.index = btc_sp500_data.index.strftime('%Y-%m-%d')

    return btc_sp500_data


def load_roberta_model():

    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # You can change this to any 3-class sentiment model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer


def load_prophet_model():
    # Load the model with joblib
    loaded_model = joblib.load('weights/prophet_model.joblib')

    return loaded_model


def sentiment_classification(text):
    if pd.isna(text):
        return 0
    
    # Tokenize the input text
    inputs = st.session_state.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = st.session_state.model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities for each class
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    finale_probability = probabilities[0][2].item() - probabilities[0][0].item()

    # finale_probability is in the range [-1, +1] 
    return finale_probability






# Main streamlit page
def main():
    # Set layout as wide
    st.set_page_config(layout="wide")
    
    # Define the title
    st.title("✨ Stock Price forecasting Dashboard")


    # get current date
    current_date = datetime.now()

    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer =  load_roberta_model()

    if 'prophet' not in st.session_state:
        st.session_state.prophet_model=load_prophet_model()      

    if 'df_news' not in st.session_state:
        # get df_news
        st.session_state.df_news = get_historical_news(current_date)

    if 'btc_sp500_data' not in st.session_state:
        # get btc and sp500 prices
        st.session_state.btc_sp500_data = get_historical_prices(current_date)

    if 'merged_data' not in st.session_state:
        # merge both btc, sp500 features and news
        st.session_state.merged_data  = pd.merge(st.session_state.btc_sp500_data, st.session_state.df_news, left_index = True, right_index = True, how = "left")

        # analyse news: negative, positive, neutral and empty
        st.session_state.merged_data["news_analysis"] = st.session_state.merged_data["Title"].apply(lambda x: sentiment_classification(x))

        # Calculate moving averages
        st.session_state.merged_data['SMA_50'] = st.session_state.merged_data['Close_BTC'].rolling(window=50).mean()
        st.session_state.merged_data['EMA_50'] = st.session_state.merged_data['Close_BTC'].ewm(span=50, adjust=False).mean()

        # reset index
        st.session_state.merged_data.reset_index(inplace=True)

        # convert Date column from str to date
        st.session_state.merged_data['Date'] = st.session_state.merged_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

        # select only date, BTC close price, S&P500 close price and the output of the news analysis function
        st.session_state.merged_data = st.session_state.merged_data[["Date", "Close_BTC", "Close_S&P500", "SMA_50", "EMA_50", "news_analysis"]]

        
        # Define target and date features
        target = "Close_BTC"
        date = "Date"

        # Rename columns columns
        st.session_state.merged_data.rename(columns={date:'ds', target:'y'}, inplace = True)


        print("columns --- : ", st.session_state.merged_data.columns)






    # Define x_min, x_max, y_min and y_max
    x_min, x_max, y_min, y_max = st.session_state.merged_data['ds'].min(), st.session_state.merged_data['ds'].max(), st.session_state.merged_data['y'].min(), st.session_state.merged_data['y'].max()


    # Initialize a global variable in session_state
    if 'new_invested_amount' not in st.session_state:
        # Initialize new invested amount with 0 
        st.session_state.new_invested_amount = 0  

    if 'investment_amount' not in st.session_state:
        # Initialize total invested amount with 0 
        st.session_state.investment_amount = 0   

    if 'nb_units' not in st.session_state:
        # Initialise number of stocks with 0 
        st.session_state.nb_units = 0

    if 'new_stock_price' not in st.session_state:
        # Initialise new stock price with 0 
        st.session_state.new_stock_price = 0

    if 'purchase_price' not in st.session_state:
        # Initialise purchase price with 0 
        st.session_state.purchase_price = 0

    if 'df_predictions' not in st.session_state:
        # Initialise df_predictions to store predictions
        st.session_state.df_predictions = pd.DataFrame(columns = [date, "y_true", "y_pred"])

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
            st.markdown("""
                <style>
                    .stButton > button {
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 80%;  /* You can adjust the width as needed */
                    }
                </style>
            """, unsafe_allow_html=True)

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

    # If one of the button is pressed
    if model_arima or model_lstm or model_prophet:
        # Spinner for long operations
        with st.spinner('Processing...'):
            for train_size in range(10, len(st.session_state.merged_data), 30):
                # Define train and test datasets
                data_train, data_test = st.session_state.merged_data[:train_size], st.session_state.merged_data[train_size:] 
                
                # If ARIMA was selected
                if model_arima:
                    # Get forecasted values using ARIMA model
                    data_train, data_test, forecasted_values = arima_model(data_train, data_test, 'ds', 'y', retrain=False)

                # if LSTM was selected
                if model_lstm:
                    # Get forecasted values using LSTM model
                    data_train, data_test, forecasted_values = lstm_model(data_train, data_test, 'ds', 'y', retrain=False)

                # if PROPHET was selected
                if model_prophet:
                    # Predict future values
                    forecast = st.session_state.prophet_model.predict(data_test[['ds']])

                    # Extract the forecasted values and dates
                    forecasted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                    if mean_squared_error(data_test['y'], forecasted_values['yhat']) > 1000:
                        print('start retraining ... ')
                        # retrain prophet model
                        train_prophet_model(data_train, data_test, 'ds', 'y')

                        # fetch the last version of the model
                        model_uri = "models:/prophet_model/latest"  
                        st.session_state.prophet_model = mlflow.prophet.load_model(model_uri)
                        print('last prophet model is fetched ... ')

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

                    # Show Current price in colored cell
                    error_placeholder.markdown(f"""
                    <div style="background-color: {"#8B0000"}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 1% auto;">
                        <strong></strong><br>
                        <strong></strong> Mean Absolute Error (MAE) of the model in $:<br>
                        <strong>{model_error:.2f}</strong><br>
                        <strong></strong><br>
                    </div>
                    """, unsafe_allow_html=True)

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
                    st.session_state.purchase_price = data_train['y'].iloc[-1]
                    st.session_state.nb_units += st.session_state.new_invested_amount/st.session_state.purchase_price
                    st.session_state.new_stock_price = st.session_state.investment_amount/st.session_state.nb_units 
                    st.session_state.invested = False
                    
                    
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
                price_placeholder.markdown(f"""
                <div style="background-color: {color_price}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 1% auto;">
                    <strong>Current price: $ {current_price:.2f}</strong><br>
                </div>
                """, unsafe_allow_html=True)

                # Show Portfolio details in colored cell
                portfolio_placeholder.markdown(f"""
                <div style="background-color: {portfolio_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 0 auto;">
                    <strong>Invested amount: $ {st.session_state.investment_amount:.2f}</strong><br>
                    <strong>Purchace price: $ {st.session_state.purchase_price:.2f}</strong><br>
                    <strong>Number of stocks:  {st.session_state.nb_units:.2f}</strong><br>
                    <strong>New stock price: $ {st.session_state.new_stock_price:.2f}</strong><br>
                </div>
                """, unsafe_allow_html=True)

                # Show P&L in colored cell
                plot_model_perf.markdown(f"""
                <div style="background-color: {color_pnl}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 0 auto;">
                    <strong>ROI: {roi:.2f}%</strong><br>
                    <strong>P&L: $ {pnl:.2f}</strong><br>
                </div>
                """, unsafe_allow_html=True)


                # Figure to plot train, test and forecasted values 
                fig = go.Figure()

                # Add training data
                fig.add_trace(go.Scatter(x=data_train['ds'], y=data_train['y'], mode='lines', name='Train Data'))

                # Add simple moving average 
                fig.add_trace(go.Scatter(x=data_train['ds'], y=data_train['SMA_50'], mode='lines', name='Simple moving average'))

                # Add Exponential Moving Average
                fig.add_trace(go.Scatter(x=data_train['ds'], y=data_train['EMA_50'], mode='lines', name='Exponential Moving Average'))

                # Add test data
                fig.add_trace(go.Scatter(x=data_test['ds'], y=data_test['y'], mode='lines', name='Test Data'))

                # Add forecasted data
                fig.add_trace(go.Scatter(x=forecasted_values['ds'], y=forecasted_values['yhat'], mode='lines', name='Forecasted Data'))

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
                    #width=800,
                    #height=800, 
                    template="plotly",
                )
                # Update the plot in Streamlit
                plot_placeholder.plotly_chart(fig)


                # Figure to plot true and predicted values
                fig_1 = go.Figure()
                
                # Add training data
                fig_1.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Forecasted price'))
                fig_1.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Real price'))

                # Layout customization
                fig_1.update_layout(
                    title='Real and forecasted prices',
                    xaxis_title='Date',
                    yaxis_title='Values',
                    xaxis=dict(range=[x_min, x_max]),  
                    #yaxis=dict(range=[y_min, 1.5*y_max]), 
                    #width=800,
                    #height=800, 
                    template="plotly",
                )
                # Update the plot in Streamlit
                plot_perf_model.plotly_chart(fig_1)


            # save date, true value and predicted value
            st.session_state.df_predictions["ds"] = dates
            st.session_state.df_predictions["y_true"] = y_true
            st.session_state.df_predictions["y_pred"] = y_pred




if __name__ == "__main__":
    main()