# app/Dockerfile

# Specify the base image
# FROM python:3.10
FROM python:3.10-slim


# Set environment variables for UTF-8 support
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Update and install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# install pip and torch   
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Step 3: Create the directory to store MLflow experiments
#RUN mkdir -p /app/mlruns

# Copy only the requirements file first to cache dependencies layer
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

# copy all elements
COPY . /app

# listen to port 8501
EXPOSE 8501

# Healthcheck command to ensure the container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# run the streamlit run command
ENTRYPOINT ["streamlit", "run", "./app/price_forecasting_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
