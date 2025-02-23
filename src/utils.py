import yfinance as yf
import praw
from datetime import datetime
import pandas as pd
import logging
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import subprocess


# API constants for fetching data
CLIENT_ID = "rVofejEZJ37RP4Ht0Di9dg"
CLIENT_SECRET = "KoytCCII7wPYlSY0Hf7jUpZ_yPn59g"



def get_historical_news(current_date):
    """
    Fetches historical Reddit posts within a given date range and returns them as a pandas DataFrame.

    Parameters:
        current_date (datetime): The current date for filtering the news.

    Returns:
        pd.DataFrame: A DataFrame containing the news data.
    """
    try :
        # Initialize Reddit API client with your credentials
        reddit = praw.Reddit(
            client_id=CLIENT_ID,  
            client_secret=CLIENT_SECRET,  
            user_agent='BTC News Scraper v1'  
        )

        # Define the date range (2020 to now)
        start_date = datetime(2019, 1, 1)

        # data news
        data = []

        # Subreddit for Bitcoin news
        subreddit = reddit.subreddit('Bitcoin')

        # Fetch posts in chunks
        for submission in subreddit.top(time_filter='all', limit=100000):  
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
    
    except Exception as e:
        logging.error(f"Error in get_historical_news: {e}")
        st.session_state.app_error_count.inc() 
        return None 



def get_historical_prices(current_date):
    """
    Fetches historical prices for Bitcoin and S&P 500 and returns them in a merged DataFrame.

    Parameters:
        current_date (str): The end date for fetching the data.

    Returns:
        pd.DataFrame: Merged DataFrame with Bitcoin and S&P 500 prices.
    """
    try:
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
    
    except Exception as e:
        logging.error(f"Error in get_historical_prices: {e}")
        st.session_state.app_error_count.inc() 
        return None 
    

def save_to_dvc(list_paths):
    """
    Save files to DVC (Data Version Control) and Git, and push the changes to DVC remote storage.
    
    Parameters:
        list_paths: list of file paths to be saved and versioned with DVC
    """
    # save the file using DVC
    for file_path in list_paths:
        try:

            # Use DVC Python API to add file
            subprocess.run(['dvc', 'add', file_path], check=True)
            
            # Stage the .dvc file for Git
            subprocess.run(['git', 'add', f'{file_path}.dvc'], check=True)
            
            # Commit the changes to Git with a dynamic commit message
            subprocess.run(['git', 'commit', '-m', f'Add {file_path}.dvc to DVC'], check=True)


        except subprocess.CalledProcessError as e:
                logging.error(f"Error in DVC or Git command: {e}")
                raise
            
    # push datasets to dvc remote storage 
    try:
        subprocess.run(['dvc', 'push'], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error pushing to DVC remote storage: {e}")
        raise
    
