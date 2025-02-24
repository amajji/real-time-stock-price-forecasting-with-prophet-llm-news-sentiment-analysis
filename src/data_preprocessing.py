from src.utils import get_historical_news, get_historical_prices
import pandas as pd
import logging
import torch
from src.sentiment_analysis import SentimentAnalysis 
from datetime import datetime

class DataPreprocessing():
    def __init__(self, current_date, sentiment_analyzer: SentimentAnalysis):
        self.current_date = current_date
        self.df_news = None
        self.df_btc_sp500_data = None
        self.df_merged_data = None
        self.sentiment_analyzer = sentiment_analyzer

    def load_data(self):
        # Load historical news and prices
        self.df_news = get_historical_news(self.current_date)
        self.df_btc_sp500_data = get_historical_prices(self.current_date)

        logging.info("Shape df_news --- : ", self.df_news.shape)
        logging.info("Shape df_btc_sp500_data --- : ", self.df_btc_sp500_data.shape)
    
        # Load the sentiment model (only once)
        self.sentiment_analyzer.load_roberta_model() 

    def merge_data(self):
        # Merge BTC & S&P500 prices with news data
        self.df_merged_data = pd.merge(self.df_btc_sp500_data, self.df_news, left_index = True, right_index = True, how = "left")

    def processing_data(self):
        # analyse news: negative, positive, neutral and empty
        self.df_merged_data["news_analysis"] = self.df_merged_data["Title"].apply(lambda x: self.sentiment_analyzer.sentiment_classification(x))

        # Calculate moving averages
        self.df_merged_data['SMA_50'] = self.df_merged_data['Close_BTC'].rolling(window=50).mean()
        self.df_merged_data['EMA_50'] = self.df_merged_data['Close_BTC'].ewm(span=50, adjust=False).mean()

        # reset index
        self.df_merged_data.reset_index(inplace=True)

        # convert Date column from str to date
        self.df_merged_data['Date'] = self.df_merged_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

        # select only date, BTC close price, S&P500 close price and the output of the news analysis function
        self.df_merged_data = self.df_merged_data[["Date", "Close_BTC", "Close_S&P500", "SMA_50", "EMA_50", "news_analysis"]]

        
        # Define target and date features
        target = "Close_BTC"
        date = "Date"

        # Rename columns columns
        self.df_merged_data.rename(columns={date:'ds', target:'y'}, inplace = True)
        logging.info("columns --- : ", self.df_merged_data.columns)
        logging.info("Shape --- : ", self.df_merged_data.shape)

