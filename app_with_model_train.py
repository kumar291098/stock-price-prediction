import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow.keras.callbacks import Callback

# Define default values for inputs
DEFAULT_START_DATE = '2010-01-01'
DEFAULT_END_DATE = '2024-01-01'
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_RATIO = 0.7

# Streamlit app layout
st.title("📈 Real-Time Stock Price Prediction")
st.sidebar.header("Model Configuration")

# User inputs in the sidebar
start_date = st.sidebar.date_input("Select Start Date", value=pd.to_datetime(DEFAULT_START_DATE))
end_date = st.sidebar.date_input("Select End Date", value=pd.to_datetime(DEFAULT_END_DATE))
epochs = st.sidebar.slider("Select Number of Epochs", min_value=10, max_value=100, value=DEFAULT_EPOCHS)
train_ratio = st.sidebar.slider("Select Train-Test Split Ratio", min_value=0.1, max_value=0.9, value=DEFAULT_TRAIN_RATIO)
stock_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value='AAPL')

# Class for Epoch Progress Callback
class StreamlitCallback(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_bar = st.progress(0)  # Initialize progress bar in Streamlit
        self.epoch_text = st.empty()     # Text placeholder for showing epoch count

    def on_epoch_end(self, epoch, logs=None):
        # Update the progress bar and text with the current epoch progress
        current_progress = int((epoch + 1) / self.total_epochs * 100)
        self.epoch_bar.progress(current_progress)
        self.epoch_text.text(f"Epoch {epoch + 1}/{self.total_epochs} completed.")

# Apply Button
if st.sidebar.button("Apply"):
    # Retrieve stock data for the specified ticker from Yahoo Finance
    df = yf.download(stock_ticker, start=start_date, end=end_date)

    # Check if the DataFrame is empty before processing
    if not df.empty:
        # Display the raw data
        st.subheader(f"Raw Stock Data for {stock_ticker}")
        st.write(df.head())  # Show the first few rows

        # Data Cleaning
        df_cleaned = df.drop(['Adj Close'], axis=1)  # Remove 'Adj Close' column
        st.subheader("Cleaned Data (Removed 'Adj Close' column)")
        st.write(df_cleaned.head())

        # Reset index to prepare data for training
        df_cleaned = df_cleaned.reset_index()

        # Data Analysis - Descriptive Statistics
        st.subheader("Descriptive Statistics of Cleaned Data")
        st.write(df_cleaned.describe())

        # Data Visualization - Closing Price Trend
        st.subheader("📉 Closing Price Trend")
        st.line_chart(df_cleaned['Close'])

        # Moving Averages
        ma100 = df_cleaned['Close'].rolling(100).mean()
        ma200 = df_cleaned['Close'].rolling(200).mean()

        # Plot with moving averages
        st.subheader("Closing Price with 100 & 200-Day Moving Averages")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_cleaned['Close'], label='Closing Price', color='blue')
        ax.plot(ma100, label='100-Day MA', color='orange')
        ax.plot(ma200, label='200-Day MA', color='red')
        ax.set_title(f"{stock_ticker} Closing Price and Moving Averages")
        ax.legend()
        st.pyplot(fig)

        # Data Preprocessing for Model Training
        data_training = pd.DataFrame(df_cleaned['Close'][0:int(len(df_cleaned)*train_ratio)])
        data_testing = pd.DataFrame(df_cleaned['Close'][int(len(df_cleaned)*train_ratio):int(len(df_cleaned))])

        # Show training and testing datasets
        st.subheader("Training Dataset Preview")
        st.write(data_training.describe())

        st.subheader("Testing Dataset Preview")
        st.write(data_testing.describe())

        # MinMax Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        st.subheader("Scaled Training Data (First 10 values)")
        st.write(data_training_array[:10])  # Display the first 10 scaled values

        # Prepare training data
        x_train, y_train = [], []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        st.subheader("Training Data Shapes")
        st.write(f"x_train shape: {x_train.shape}")
        st.write(f"y_train shape: {y_train.shape}")

        # Model Building and Summary
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))

        # Display the model summary
        st.subheader("Model Summary")
        st.text(model.summary())

        # Compile Model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training Progress with Callback
        st.subheader("Training Progress")
        training_callback = StreamlitCallback(total_epochs=epochs)
        history = model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[training_callback])

        st.success("Model training completed!")

        # Prepare Testing Data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Model Prediction
        y_predicted = model.predict(x_test)

        # Scale back the predicted values
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Final Prediction vs Original Plot
        st.subheader('📊 Prediction vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)
    else:
        st.warning("⚠️ No data found for the provided ticker symbol.")
