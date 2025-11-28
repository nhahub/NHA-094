# Use TensorFlow base image to avoid slow download
FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --ignore-installed -r requirements.txt

# Copy application files
COPY stock_predictor.py .
COPY app.py .
COPY gemini_advisor.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
