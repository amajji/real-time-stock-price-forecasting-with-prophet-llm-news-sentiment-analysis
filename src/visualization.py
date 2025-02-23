import plotly.graph_objects as go
import streamlit as st


class Visualization():
    def __init__(self):
        pass


    def plot_current_price(self, color_price, current_price):
        return f"""
                <div style="background-color: {color_price}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 1% auto;">
                    <strong>Current price: $ {current_price:.2f}</strong><br>
                </div>
                """


    def plot_portfolio_details(self, portfolio_color):
        return f"""
                <div style="background-color: {portfolio_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 0 auto;">
                    <strong>Invested amount: $ {st.session_state.investment_amount:.2f}</strong><br>
                    <strong>Purchace price: $ {st.session_state.purchase_price:.2f}</strong><br>
                    <strong>Number of stocks:  {st.session_state.nb_units:.2f}</strong><br>
                    <strong>New stock price: $ {st.session_state.new_stock_price:.2f}</strong><br>
                </div>
                """


    def plot_buy_button(self):
        return """
                <style>
                    .stButton > button {
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 80%;  /* You can adjust the width as needed */
                    }
                </style>
            """


    def plot_pnl(self, pnl, roi):
        color = 'green' if pnl > 0 else 'red'
        return f"""
        <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <strong>ROI: {roi:.2f}%</strong><br>
            <strong>P&L: ${pnl:.2f}</strong><br>
        </div>
        """


    def plot_forecast(self, data_train, data_test, forecasted_values):
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

        return fig
    

    def plot_true_forecasted_values(self, dates, y_true, y_pred, x_min, x_max):
        # Figure to plot true and predicted values
        fig = go.Figure()
        
        # Add training data
        fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Forecasted price'))
        fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Real price'))

        # Layout customization
        fig.update_layout(
            title='Real and forecasted prices',
            xaxis_title='Date',
            yaxis_title='Values',
            xaxis=dict(range=[x_min, x_max]),  
            #yaxis=dict(range=[y_min, 1.5*y_max]), 
            #width=800,
            #height=800, 
            template="plotly",
        )
        return fig


    

    
    

    

    

    
    def plot_model_mae(self, model_error):
        return f"""
                    <div style="background-color: {"#8B0000"}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 1% auto;">
                        <strong></strong><br>
                        <strong></strong> Mean Absolute Error (MAE) of the model in $:<br>
                        <strong>{model_error:.2f}</strong><br>
                        <strong></strong><br>
                    </div>
                    """
    
