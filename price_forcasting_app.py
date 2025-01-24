import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from prophet import Prophet
import joblib
import plotly.graph_objects as go
import pandas as pd
import openpyxl
import matplotlib.colors as mcolors

def forecaste(merged_data, train_size):

    data_train, data_test = merged_data[:train_size], merged_data[train_size:]

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
    prophet_model.fit(data_train)

    # Predict future values
    forecast = prophet_model.predict(data_test[['ds', 'y']])

    # Extract the forecasted values and dates
    forecasted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return data_train, data_test, forecasted_values



#########################################################################################
#                                Main code                                              #
#########################################################################################


# First streamlit's page
def page_1():

    
    # define the title
    st.title("✨ Price forcasting with Prophet")

    # # quick decription of the webapp
    # st.markdown(
    #     """
    #     This interactive dashboard allows users to extract information.

    #     """
    # )


    # Fetch cryptocurrency data (e.g., Bitcoin)
    btc_data = yf.download('BTC-USD', start='2019-01-01', end='2025-01-01')

    # Assume you have another dataset, such as S&P 500
    #sp500_data = yf.download('^GSPC', start='2019-01-01', end='2025-01-01')

    # get only level 0 columns
    btc_data.columns = btc_data.columns.levels[0]
    #sp500_data.columns = sp500_data.columns.levels[0]

    # merge sp500_data with btc_data
    #merged_data  = pd.merge(btc_data, sp500_data, left_index = True, right_index = True, suffixes=('_BTC', '_S&P500'))
    merged_data = btc_data

    # reset index of train data
    merged_data = merged_data.reset_index()

    print("columns -- ", merged_data.columns)



    # Example candlestick chart with OHLC data
    # fig = go.Figure(data=[go.Candlestick(x=merged_data['Date'],
    #                 open=merged_data['Open_BTC'], high=merged_data['High_BTC'],
    #                 low=merged_data['Low_BTC'], close=merged_data['Close_BTC'],
    #                 name='Candlestick Chart')])

    # fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    # st.plotly_chart(fig)


    # for prediction, we will only keep BTC and S&P 500 closing prices
    #merged_data = merged_data[["Date", "Close_BTC", "Close_S&P500"]]
    merged_data = merged_data[["Date", "Close"]]



    # rename train's columns
    merged_data.rename(columns={'Date':'ds', 'Close':'y'}, inplace = True)

    # Calculate moving averages
    merged_data['SMA_50'] = merged_data['y'].rolling(window=50).mean()
    merged_data['EMA_50'] = merged_data['y'].ewm(span=50, adjust=False).mean()



    # Split data into train and test sets (80% train, 20% test)
    # train_size = int(len(merged_data) * 0.8)
    #train_size = int(len(merged_data) * 0.5)
    # Plot the train, test, and forecasted data


    # define x_min, x_max, y_min and y_max
    x_min, x_max, y_min, y_max = merged_data["ds"].min(), merged_data["ds"].max(), merged_data["y"].min(), merged_data["y"].max()
    


    # Initialize a global variable in session_state
    if 'new_invested_amount' not in st.session_state:
        st.session_state.new_invested_amount = 0

    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 0

    if 'nb_units' not in st.session_state:
        st.session_state.nb_units = 0

    if 'new_stock_price' not in st.session_state:
        st.session_state.new_stock_price = 0

    if 'purchase_price' not in st.session_state:
        st.session_state.purchase_price = 0


    if 'df_predictions' not in st.session_state:
        st.session_state.df_predictions = pd.DataFrame(columns = ["ds", "y_true", "y_pred"])

    # define 2 columns 
    col1, col2, col3 = st.columns(3)

    with col1:
        # Create a container for dynamic plot updates
        portfolio_placeholder = st.empty()

    with col2:
        # Create a container to show the current price
        price_placeholder = st.empty()
        # Create a container for dynamic plot updates
        plot_model_perf = st.empty()

    with col3:
        error_placeholder = st.empty()


    # define 2 columns 
    col4, col5 = st.columns(2)

    with col4:
        # Create a container for dynamic plot updates
        plot_placeholder = st.empty()

        col6, col7 = st.columns(2)

        with col6:
            # Create date input widgets for start and end date selection
            start_date = st.date_input(
                'Select start date', 
                min_value=merged_data['ds'].min(), 
                max_value=merged_data['ds'].max(), 
                value=merged_data['ds'].min())
        
        with col7:
            end_date = st.date_input(
                'Select end date', 
                min_value=merged_data['ds'].min(), 
                max_value=merged_data['ds'].max(), 
                value=merged_data['ds'].max())

    with col5:
        plot_perf_model = st.empty()


        # define col 5 an 6
        col8, col9 = st.columns(2)

        with col8:

            # Input for amount to invest
            st.session_state.new_invested_amount = st.number_input('Enter the amount you want to invest:',min_value=0, step=1)
            st.session_state.investment_amount += st.session_state.new_invested_amount 
            
        with col9:
            #st.write("Click on the the button to buy")

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



        
    merged_data = merged_data.query("@start_date<=ds and ds<=@end_date")

    pnl, roi = 0, 0 
    bool_press, bool_first_time = False, True
    color_price, portfolio_color = 'gray', 'darkblue'
    y_pred, y_true, dates = [], [], []
    #with col4:

    # Spinner for long operations
    with st.spinner('Processing...'):
        for train_size in range(10, len(merged_data), 10):
            data_train, data_test, forecasted_values = forecaste(merged_data, train_size)
            # current price 
            current_price = data_test['y'].iloc[0]


            if not bool_first_time: 
                y_pred.append(next_predicted_value)
                y_true.append(data_train['y'].iloc[-1])
                dates.append(data_train['ds'].iloc[-1])
                # Calculate MAE
                model_error = mean_absolute_error(data_test['y'], forecasted_values['yhat'])


                cmap = plt.get_cmap("YlOrRd")  # Yellow to Red (dark red for high values)
                norm = mcolors.Normalize(vmin=0, vmax=60000)  # Define your MAE range, adjust vmax as needed

                # Get color corresponding to the MAE value
                color_error = mcolors.to_hex(cmap(norm(model_error)))


                # define the color depending on the variation between old and new stock price
                # if model_error >= 0:
                #     color_error = 'green'
                # else:
                #     color_error = 'red'

                # Show Current price in colored cell
                error_placeholder.markdown(f"""
                <div style="background-color: {"#8B0000"}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 1% auto;">
                    <strong></strong><br>
                    <strong></strong> Mean Absolute Error (MAE) of the model in $:<br>
                    <strong>{model_error:.2f}</strong><br>
                    <strong></strong><br>
                </div>
                """, unsafe_allow_html=True)



                # define the color depending on the variation between old and new stock price
                if current_price >= old_value:
                    color_price = 'green'
                else:
                    color_price = 'red'
                # set the new price as old value
                old_value = current_price

            # predicted value
            next_predicted_value = forecasted_values['yhat'].iloc[0]

            if bool_first_time:
                old_value = current_price

            bool_first_time = False

            if st.session_state.invested:
                bool_press = True
                st.session_state.purchase_price = data_train['y'].iloc[-1]
                st.session_state.nb_units += st.session_state.new_invested_amount/st.session_state.purchase_price
                st.session_state.new_stock_price = st.session_state.investment_amount/st.session_state.nb_units 
                st.session_state.invested = False
                
                
            
            if bool_press: 
                pnl = (current_price - st.session_state.new_stock_price)*st.session_state.nb_units
                roi = 100*pnl/st.session_state.investment_amount

            # define the colors depending on the sign of the pnl
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
                <strong>Purchaced price: $ {st.session_state.purchase_price:.2f}</strong><br>
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

            fig_1 = go.Figure()

            # Add training data
            fig_1.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Forcasted price'))
            fig_1.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Real price'))

            # Layout customization
            fig_1.update_layout(
                title='Real and forcasted prices',
                xaxis_title='Date',
                yaxis_title='Values',
                xaxis=dict(range=[x_min, x_max]),  
                #yaxis=dict(range=[y_min, 1.5*y_max]), 
                #width=800,
                #height=800, 
                template="plotly",
            )

            # # Update the plot in Streamlit
            plot_perf_model.plotly_chart(fig_1)


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

        st.session_state.df_predictions["ds"] = dates
        st.session_state.df_predictions["y_true"] = y_true
        st.session_state.df_predictions["y_pred"] = y_pred


        st.session_state.df_predictions.to_excel("performace1.xlsx")











def main():

    st.set_page_config(layout="wide")


    st.sidebar.title("Menu")

    PAGES = {
        "🎈 LLaMA2-7B LLM": page_1,
    }

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Select your page : ", list(PAGES.keys()))
    #st.sidebar_caption()

    PAGES[selection]()

    st.sidebar.title("About")
    st.sidebar.info(
        """
    Web App URL: <https://amajji-streamlit-dash-streamlit-app-8i3jn9.streamlit.app/>
    GitHub repository: <https://github.com/amajji/LLM-RAG-Chatbot-With-LangChain>
    """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    MAJJI Anass 
    [GitHub](https://github.com/amajji) | [LinkedIn](https://fr.linkedin.com/in/anass-majji-729773157)
    """
    )


if __name__ == "__main__":
    main()