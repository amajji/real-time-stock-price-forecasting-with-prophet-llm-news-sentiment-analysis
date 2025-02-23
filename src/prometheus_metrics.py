import streamlit as st
from streamlit_extras import prometheus
from prometheus_client import start_http_server, Gauge, Counter, Histogram, REGISTRY


class PrometheusMetrics():
    def __init__(self):
        # Define the Prometheus metrics
        if 'app_error_count' not in st.session_state:
            st.session_state.app_error_count = Counter(
                name='app_exception_count', 
                documentation='Number of application errors', 
                registry=prometheus.streamlit_registry()
                )

            if 'prophet_model_retrain_count' not in st.session_state:
                st.session_state.prophet_model_retrain_count = Counter(
                    name='prophet_model_retrain_count', 
                    documentation='Number of retrains of Prophet model', 
                    registry=prometheus.streamlit_registry()
                    )

            if 'prophet_model_training_duration' not in st.session_state:
                st.session_state.prophet_model_training_duration = Histogram(
                    name='prophet_model_training_duration_seconds', 
                    documentation='Time taken for Prophet model training', 
                    registry=prometheus.streamlit_registry()
                    )

            if 'request_duration' not in st.session_state:
                st.session_state.request_duration = Histogram(
                    name='request_duration_seconds', 
                    documentation='Time taken for request processing', 
                    registry=prometheus.streamlit_registry()
                    )

            if 'model_error_rate' not in st.session_state:
                st.session_state.model_error_rate = Gauge(
                    name='model_error_rate', 
                    documentation='Error rate of the model predictions', 
                    registry=prometheus.streamlit_registry()
                    )

