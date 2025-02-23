import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch


class SentimentAnalysis():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_model_loaded = False

    def load_roberta_model(self):
        """
        Loads the RoBERTa model for sentiment classification.
        """
        if self.is_model_loaded:
            logging.info("Model already loaded.")
            return self.model, self.tokenizer
        
        try:
            # Load model and tokenizer
            model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # You can change this to any 3-class sentiment model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.is_model_loaded = True
            return self.model, self.tokenizer
            
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            st.session_state.app_error_count.inc() 
            return None, None
        
    def sentiment_classification(self, text):
        """
        Classifies sentiment of the input text.

        Parameters:
            text (str): The text to classify.

        """
        try:
            if pd.isna(text):
                return 0
            
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Apply softmax to get probabilities for each class
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            finale_probability = probabilities[0][2].item() - probabilities[0][0].item()

            # finale_probability is in the range [-1, +1] 
            return finale_probability
        
        except Exception as e:
            logging.error(f"Error in sentiment_classification: {e}")
            st.session_state.app_error_count.inc() 
            return None 