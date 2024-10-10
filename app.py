import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Set the title of the Streamlit app
st.set_page_config(page_title="Stock Trend Prediction", page_icon="📈")
st.title('📈 Stock Trend Prediction')

# Theme selection


# Create a text input for the user to enter the stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Define the start and end dates
start = '2010-01-01'
end = '2024-01-01'

# Retrieve stock data for the specified ticker from Yahoo Finance
df = yf.download(user_input, start=start, end=end)

# Check if the DataFrame is empty before displaying the data
if not df.empty:
    # Display a subheader and the data description
    st.subheader(f'Data from {start} to {end}')
    st.write(df.describe())
    
    # Display the raw data
    st.subheader('📊 Raw Data')
    st.write(df)

    # Plotting the Closing Price
    st.subheader('📉 Closing Price')
    st.line_chart(df['Close'])

    # Closing Price vs Time Chart
    st.subheader("Closing Price vs Time Chart")
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title(f'{user_input} Closing Price Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.grid()
    plt.legend()
    st.pyplot(fig1)

    # Closing Price with 100-Day Moving Average
    st.subheader("Closing Price vs Time Chart with 100-Day Moving Average")
    ma100 = df['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.plot(ma100, label='100-Day MA', color='orange')
    plt.title(f'{user_input} Closing Price and 100-Day MA', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.grid()
    plt.legend()
    st.pyplot(fig2)

    # Closing Price with 100 and 200-Day Moving Averages
    st.subheader("Closing Price vs Time Chart with 100 & 200-Day Moving Averages")
    ma200 = df['Close'].rolling(200).mean()
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.plot(ma100, label='100-Day MA', color='orange')
    plt.plot(ma200, label='200-Day MA', color='red')
    plt.title(f'{user_input} Closing Price, 100-Day and 200-Day MAs', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.grid()
    plt.legend()
    st.pyplot(fig3)

    # Preparing data for LSTM
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load the LSTM model
    model = load_model('Stock_price_prediction_keras_LSTM_model.h5')

    # Testing part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predict = model.predict(x_test)
    
    # Rescale predictions
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predict * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader('📈 Prediction vs Original')
    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.title('Prediction vs Original', fontsize=16)
    plt.legend()
    st.pyplot(fig4)

else:
    st.warning("⚠️ No data found for the provided ticker symbol.")

# Additional project information
st.sidebar.header("About This Project")
st.sidebar.info(
    """
    This Streamlit app predicts stock trends using historical data. 
    Users can input a stock ticker (e.g., AAPL, TSLA) to visualize its closing prices and predictions.
    
    **Features:**
    - Data visualization with moving averages
    - Predictions using LSTM models

    **How to Use:**
    1. Enter the stock ticker in the input box.
    2. View the visualizations and predictions.
    """
)
