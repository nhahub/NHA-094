import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from stock_predictor import StockPricePredictor, ALL_TICKERS, set_seeds
from gemini_advisor import get_gemini_advice

# Page configuration
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'history' not in st.session_state:
    st.session_state.history = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Header
st.title("📈 LSTM Stock Price Predictor")
st.markdown("### Interactive Multi-Horizon Stock Forecasting Dashboard")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker selection
    st.subheader("📊 Stock Selection")
    selected_ticker = st.selectbox(
        "Choose a ticker:",
        options=ALL_TICKERS,
        index=ALL_TICKERS.index('AAPL') if 'AAPL' in ALL_TICKERS else 0
    )
    
    # Hardcoded parameters (hidden from UI)
    period = '3y'
    interval = '1d'
    lookback = 60
    horizons = (1, 30, 90, 180)
    units = 64
    dropout = 0.2
    epochs = 30
    batch_size = 32
    learning_rate = 1e-3
    gemini_api_key = "AIzaSyAPwT7N716JovCHACF8D-mhemIZU_odxxU"

    st.markdown("---")
    
    # Action buttons
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        st.session_state.trained = False
        st.session_state.predictions = None
        
        with st.spinner("🔄 Initializing predictor..."):
            set_seeds(42)
            predictor = StockPricePredictor(
                symbol=selected_ticker,
                lookback=lookback,
                horizons=horizons
            )
            st.session_state.predictor = predictor
        
        try:
            # Fetch data
            with st.spinner(f"📥 Fetching data for {selected_ticker}..."):
                predictor.fetch_data(period=period, interval=interval)
                st.success(f"✅ Fetched {len(predictor.data)} data points")
            
            # Preprocess
            with st.spinner("🔧 Preprocessing data..."):
                Xtr, Ytr, Xte, Yte = predictor.preprocess(train_ratio=0.8)
                st.session_state.test_data = (Xte, Yte)
            
            # Build model
            with st.spinner("🏗️ Building LSTM model..."):
                predictor.build(units=units, dropout=dropout, lr=learning_rate, tau=0.6)
            
            # Train model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎓 Training model..."):
                history = predictor.train(
                    Xtr, Ytr,
                    epochs=epochs,
                    batch_size=batch_size,
                    val_split=0.1,
                    verbose=0
                )
                st.session_state.history = history
                progress_bar.progress(100)
                status_text.success("✅ Training complete!")
            
            # Evaluate
            with st.spinner("📊 Evaluating model..."):
                Y_hat_real, Y_test_real, metrics = predictor.evaluate(Xte, Yte)
                st.session_state.test_predictions = (Y_hat_real, Y_test_real)
                st.session_state.metrics = metrics
            
            # Get predictions
            with st.spinner("🔮 Generating predictions..."):
                predictions = predictor.multi_horizon_prices()
                st.session_state.predictions = predictions
            
            st.session_state.trained = True
            st.success("🎉 Model trained successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.trained = False
    
    if st.session_state.trained and st.button("🔄 Reset", use_container_width=True):
        st.session_state.trained = False
        st.session_state.predictor = None
        st.session_state.history = None
        st.session_state.predictions = None
        st.session_state.metrics = None
        st.rerun()

# Main content area
if not st.session_state.trained:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Configure parameters in the sidebar and click **Train Model** to get started!")
        
        st.markdown("### 🌟 Features")
        st.markdown("""
        - **Multi-horizon forecasting**: Predict prices for 1, 30, 90, and 180 days ahead
        - **LSTM neural network**: Advanced deep learning architecture
        - **Asymmetric loss**: Pinball loss function to reduce underestimation
        - **Bias correction**: Improved prediction accuracy
        - **Interactive visualizations**: Beautiful charts and graphs
        - **Real-time training**: Watch your model learn in real-time
        """)
        
        st.markdown("### 📚 How it works")
        st.markdown("""
        1. **Select a stock ticker** from the dropdown (e.g., AAPL, MSFT, TSLA)
        2. **Configure parameters** like lookback period and training epochs
        3. **Train the model** and watch it learn from historical data
        4. **View predictions** for multiple time horizons
        5. **Download results** as CSV for further analysis
        """)

else:
    # Results display
    predictor = st.session_state.predictor
    predictions = st.session_state.predictions
    metrics = st.session_state.metrics
    history = st.session_state.history
    
    # Header with ticker info
    st.success(f"✅ Model trained for **{selected_ticker}**")
    
    # Metrics row
    st.markdown("### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MSE", f"{metrics['mse']:.6f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.6f}")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.6f}")
    with col4:
        st.metric("MAPE", f"{metrics['mape']:.2f}%")
    
    st.markdown("---")
    
    # Predictions
    st.markdown("### 🔮 Multi-Horizon Price Predictions")
    
    # Create prediction cards
    pred_cols = st.columns(len(predictions))
    for idx, (horizon, price) in enumerate(predictions.items()):
        with pred_cols[idx]:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:1.2rem;">{horizon} Day(s)</h3>
                    <h2 style="margin:0.5rem 0 0 0; font-size:2rem;">${price:.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    
    # Tab layout for different charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Training Loss",
        "🎯 Predictions vs Actual",
        "📊 Multi-Horizon Forecast",
        "📅 Historical Prices"
    ])
    
    with tab1:
        # Training and validation loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig.update_layout(
            title=f"{selected_ticker} - Training & Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Actual vs Predicted (1-day horizon)
        Y_hat_real, Y_test_real = st.session_state.test_predictions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=Y_test_real[:, 0],
            mode='lines',
            name='Actual',
            line=dict(color='#2ca02c', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=Y_hat_real[:, 0],
            mode='lines',
            name='Predicted',
            line=dict(color='#d62728', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f"{selected_ticker} - Actual vs Predicted (1-day horizon)",
            xaxis_title="Test Sample",
            yaxis_title="Cumulative Log-Return",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Multi-horizon predictions bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[f"{k}d" for k in predictions.keys()],
                y=list(predictions.values()),
                marker=dict(
                    color=list(predictions.values()),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Price ($)")
                ),
                text=[f"${v:.2f}" for v in predictions.values()],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title=f"{selected_ticker} - Predicted Prices by Horizon",
            xaxis_title="Horizon",
            yaxis_title="Price ($)",
            height=500,
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Historical prices
        hist_data = predictor.get_historical_prices()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#9467bd', width=2),
            fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.1)'
        ))
        fig.update_layout(
            title=f"{selected_ticker} - Historical Close Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # AI Advisor Section
    st.markdown("### 🤖 AI Financial Advisor")
    
    if st.button("💡 Get AI Trading Advice", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing market data and predictions..."):
            # Get current price (last close)
            current_price = predictor.data['Close'].iloc[-1]
            
            # Generate advice
            advice = get_gemini_advice(
                api_key=gemini_api_key,
                ticker=selected_ticker,
                current_price=current_price,
                predictions=predictions
            )
            
            st.markdown(f"""
            <div style="background-color: #e6f3ff; color: #000000; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                {advice}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download section
    st.markdown("### 💾 Download Results")
    
    # Prepare CSV data
    results_df = pd.DataFrame({
        'Horizon (days)': list(predictions.keys()),
        'Predicted Price ($)': list(predictions.values())
    })
    
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(results_df, use_container_width=True)
    with col2:
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{selected_ticker}_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit and TensorFlow | LSTM Stock Price Predictor v1.0</p>
    </div>
""", unsafe_allow_html=True)
