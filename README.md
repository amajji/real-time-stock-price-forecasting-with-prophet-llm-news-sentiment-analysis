# Stock Price Forecasting and News Sentiment Analysis Dashboard
# Development of a real time price forecasting dashboard using Prophet and news sentiment analysis with LLM
Data scientist | [Anass MAJJI](https://www.linkedin.com/in/anass-majji-729773157/)
***

## :monocle_face: Description
In this project, I developed a real-time Streamlit dashboard for stock price forecasting using Prophet model for time series forecasting and news sentiment analysis using a quantized version of LLaMA with 1 billion paramters. The stock price forecasting is powered by the Prophet model, which uses both historical and real-time price data, along with the sentiment of real-time news (provided by LLaMA) as a regressor to improve predictions. The data for both historical and real time prices is fetched using the YFinance API.



1. Clone the repository
```bash
git clone https://github.com/your-username/stock-price-forecasting-sentiment-analysis.git
cd stock-price-forecasting-sentiment-analysis
```

2. Create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate 
```

3. Install required dependencies
```bash
pip install -r requirements.txt
```

4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
