import streamlit as st
import pandas as pd

class InvestmentTracker():
    def __init__(self, date):
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


    def update_investment(self, data_train):
        st.session_state.purchase_price = data_train['y'].iloc[-1]
        st.session_state.nb_units += st.session_state.new_invested_amount/st.session_state.purchase_price
        st.session_state.new_stock_price = st.session_state.investment_amount/st.session_state.nb_units 
        st.session_state.invested = False