# 📈 Stock Price Predictor

A Streamlit application for stock price prediction using LSTM neural networks, featuring multi-horizon forecasting and an AI financial advisor.

## 🚀 How to Run

You can run this application either locally with Python or using Docker.

### Option 1: Run Locally

1.  **Prerequisites**: Ensure you have Python 3.8 or higher installed.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

4.  **Access the App**: Open your browser and navigate to `http://localhost:8501`.

### Option 2: Run with Docker

1.  **Prerequisites**: Ensure you have Docker and Docker Compose installed.

2.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the App**: Open your browser and navigate to `http://localhost:8501`.

## 📂 Project Structure

- `app.py`: Main Streamlit application file.
- `stock_predictor.py`: Core logic for LSTM model and data processing.
- `gemini_advisor.py`: Module for AI-powered trading advice.
- `requirements.txt`: Python dependencies.
- `Dockerfile` & `docker-compose.yml`: Docker configuration.

## 📝 Notes

- The application uses a hardcoded API key for demonstration purposes. For production use, it is recommended to use environment variables.
