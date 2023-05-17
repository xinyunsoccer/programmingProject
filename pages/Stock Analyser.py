#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:07:56 2023

@author: lucas
"""
# import the needed libaries 
import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import requests
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
register_matplotlib_converters()

# with this API key we can query the Alpha Vantage API
API_KEY = 'YB3L9H497PDWJJ5K4'

# Set the title for the Streamlit app 
st.title("💰🚀 Stock Analyser App 🚀💰")


# Set the minimum and maximum start date values the max_date is the day before yesterday 
min_date = date(2010, 1, 1)
max_date = date.today()-timedelta(days=2)

# Get the start date from the user using a slider
START= st.slider(
    'Select a start date:',
    min_value=min_date,
    max_value=max_date,
    value=date(2015, 1, 1),
    format="YYYY-MM-DD"
)

# Print the selected start date
st.write('Start date selected:', START.strftime('%Y-%m-%d'))

# Get the current date 
TODAY = date.today().strftime('%Y-%m-%d')

# Function to get the stock ticker for the company name which the user inserted 
def get_stock_ticker(company_name):
    # Query the Alpha Vantage API to get the stock ticker info 
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'bestMatches' not in data:
         # If no matching results found, print an error message and return None
        print(f"No results found for '{company_name}'.")
        return None
    

    matches = data['bestMatches']
    if len(matches) == 0:
        # If no matches found, print an error message and return None
        print(f"No results found for '{company_name}'.")
        return None

     # Extract the best match (first match) from the response
    best_match = matches[0]
    ticker = best_match['1. symbol']
    name = best_match['2. name']
    print(f"Best Match for '{company_name}': {ticker} ({name})")

    return ticker

# Prompt the user to enter the company name (with a default value of 'Apple Inc')
company_name = st.text_input("Enter company name (If this does not work you need to enter the Stock Ticker directly)", 'Apple Inc')

# Get the stock ticker for the entered company name by calling the function
ticker = get_stock_ticker(company_name)

# Check if a valid ticker is obtained
if ticker is not None:
    selected_stock = ticker
     # Display the selected stock 
    st.write(f'Selected stock is {selected_stock}')
else:
    # Display an error message if no ticker is found for the entered company name and ask user to insert the right ticker symbol
    st.error(f"No results found for '{company_name}' please insert the ticker directly.")


# Function to load the stock data 
def load_data(ticker, START): 
    # Load historical stock data using Yahoo Finance API for the specified ticker and date range
    data = yf.download(ticker, start=START, end=TODAY, repair=True)
    data.reset_index(inplace=True)
    return data

# Display a text message indicating the loading state
data_load_state = st.text(f"Loading data for: {company_name} ({ticker}) from {START} to {TODAY}...")
# Load the stock data for the selected stock and date range
data = load_data(ticker, START) 

# Check if the loaded data is valid and sufficient
if data.shape[0] < 2 or data.isna().sum().sum() > 0:
     # Display an error message if the data is insufficient or invalid
    st.error(f"Insufficient or invalid data for: {company_name} from {START} to {TODAY}. Please try another ticker or date range.")
else:
    # Display a success message if the data loading is successful
    data_load_state.text(f'Loading data for: {company_name} from {START} to {TODAY} is done!')
    # Get Ticker object for the given ticker
    ticker_info = yf.Ticker(ticker)


st.write('------------------------------------------------------------------------')

# Section to Display a summary of the company

# Retrieve the name of the stock
stock_name = ticker_info.info['longName']

# Fetch company summary
company_summary = ticker_info.info['longBusinessSummary']

# Display the company summary
st.subheader(f'Summary of {stock_name}:')
st.write(company_summary)

st.write('------------------------------------------------------------------------')

# Section to visiualize the stock performance 

# plot the stock performance 
st.subheader(f'Stock Performance of {stock_name}:')
st.write(f'In this section you can see how {stock_name} performed over time. Moreover, you can see the opening and closing price. By moving the slider below the chart you can zoom into specific time frames.')
def plot_raw_data(): 
    # Plot the raw data, including opening and closing price 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text=f'Performance of {stock_name} stock over time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig) # Display the figure using plotly_chart()
    
plot_raw_data()

# Plot the moving averages of the stock
def calculate_moving_averages(data):
    #Calculate and plot the 50-day and 200-day moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    st.subheader('Moving Averages')
    st.write(f'In this section you can see how the moving averages of {stock_name} developed over time. By moving the slider below the chart you can zoom into specific time frames.')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name='50-day Moving Average'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], name='200-day Moving Average'))

    # Update layout with title, axes labels, and range slider visibility
    fig.update_layout(title='Moving Averages: 50-day & 200-day',
                      xaxis_title='Date',
                      yaxis_title='Price', 
                      xaxis_rangeslider_visible=True) 

    st.plotly_chart(fig) # Display the figure using plotly_chart()

calculate_moving_averages(data)

st.write('------------------------------------------------------------------------')

# Section to display KPIs of the stock
st.subheader(f'Key Performance Indicators of {stock_name}:')


# Extract the relevant KPIs
market_cap = ticker_info.info['marketCap']
pe_ratio = ticker_info.info['trailingPE']
earnings = ticker_info.info['trailingEps']
current_price = ticker_info.info['ask']
beta = ticker_info.info['beta']


# Format the numbers
formatted_market_cap = '{:,.2f}'.format(market_cap) + ' USD'
formatted_pe_ratio = '{:.2f}'.format(pe_ratio)
formatted_earnings = '{:.2f}'.format(earnings) + ' USD'
formatted_current_price = '{:.2f}'.format(current_price) + ' USD' if current_price is not None else 'N/A'
formatted_beta = '{:.2f}'.format(beta)

# Display the formatted KPIs
st.write(f"Market Cap: {formatted_market_cap}")
st.write(f"P/E Ratio: {formatted_pe_ratio}")
st.write(f"Earnings per Share: {formatted_earnings}")
st.write(f"Current Price: {formatted_current_price}")
st.write(f"Beta: {formatted_beta}")

# Retrieve major institutional holders
holders = ticker_info.institutional_holders

# Filter the top 5 major holders
top_holders = holders.head(5)

# Create a bar plot of the major holders
st.write(f'Major Holders of {stock_name}:')
chart_data = top_holders.set_index('Holder')['% Out']
st.bar_chart(chart_data)

st.write('------------------------------------------------------------------------')


# Section to plot the stock performance compared to a benchmark indice

st.subheader(f'Stock Performance of {stock_name} compared to a Benchmark Indice')
st.write(f'In this section you can see how {stock_name} performed against a Benchmark indice of your choice over a specified time horizon.')
# Benchmark Stock to an indice
benchmark_indices = ['S&P 500', 'NASDAQ', 'Dow Jones Industrial Average', 'FTSE 100', 'Nikkei 225', 'DAX', 'SIX']
selected_index = st.selectbox('Select Benchmark Index', benchmark_indices)
if selected_index == 'S&P 500':
    benchmark_ticker = '^GSPC'  # Ticker symbol for S&P 500
elif selected_index == 'NASDAQ':
    benchmark_ticker = '^IXIC'  # Ticker symbol for NASDAQ
elif selected_index == 'Dow Jones Industrial Average':
    benchmark_ticker = '^DJI'  # Ticker symbol for Dow Jones Industrial Average
elif selected_index == 'FTSE 100':
    benchmark_ticker = '^FTSE'  # Ticker symbol for FTSE 100
elif selected_index == 'Nikkei 225':
    benchmark_ticker = '^N225'  # Ticker symbol for Nikkei 225
elif selected_index == 'DAX':
    benchmark_ticker = '^GDAXI'  # Ticker symbol for DAX
elif selected_index == 'SIX':
    benchmark_ticker = '^SSMI'  # Ticker symbol for SIX

# Fetch benchmark index data
benchmark_data = yf.download(benchmark_ticker, start=START, end=TODAY)

# plot the benchmark indice against the stock data
def plot_benchmark():
    fig = go.Figure()
    
    # Add stock data
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close', yaxis='y1', line=dict(color='blue')))
    
    # Add benchmark index data
    fig.add_trace(go.Scatter(x=benchmark_data.index, y=benchmark_data['Close'], name='Benchmark Close', yaxis='y2', line=dict(color='red')))
    
    
    # Configure y-axes
    fig.update_layout(yaxis=dict(title='Stock Price', side='left', showgrid=False),
                      yaxis2=dict(title='Benchmark Index', side='right', overlaying='y', showgrid=False),  legend=dict(x=0, y=1))
    
    fig.layout.update(title_text=f'Performance of {stock_name} vs Benchmark Index', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_benchmark()

st.write('------------------------------------------------------------------------')

# section to predict the future performance of the stock 


# Prepare the data for Random Forest Regressor
def prepare_data(df):
    df = df[['Date', 'Close']].copy()
    df.set_index('Date', inplace=True)
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# Split the data into features (X) and target variable (y)
def split_data(df):
    X = df.drop('Close', axis=1)
    y = df['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# Prepare lagged features for prediction
def prepare_future_data(df, period):
    last_date = df.index[-1]
    last_close = df['Close'].iloc[-1]
    lagged_features = []
    for i in range(1, 6):
        lagged_value = df.loc[last_date - pd.Timedelta(days=i)]['Close'] if last_date - pd.Timedelta(days=i) in df.index else last_close
        lagged_features.append(lagged_value)
    future_dates = pd.date_range(start=last_date, periods=period, freq='D')
    return future_dates, lagged_features

# Make predictions for the future dates
def make_predictions(model, future_dates, lagged_features, period):
    predictions = []
    for _ in range(period):
        input_data = pd.DataFrame([lagged_features], columns=[f'Lag_{i}' for i in range(1, 6)])
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
        lagged_features.pop(0)
        lagged_features.append(prediction[0])
    return pd.DataFrame({'Date': future_dates, 'Close': predictions})

# Plot the forecasted predictions
def plot_forecast(data, forecast):
    """Plot the forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Close'], name='Forecast'))
    fig.update_layout(title='Stock Price Forecast (Random Forest Regressor)',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=True)
    fig.show()


# Load and validate the data
data = load_data(ticker, START)
assert data.shape[0] > 2, f"Insufficient data for: {company_name} from {START} to {TODAY}. Please try another ticker or date range."
assert data.isna().sum().sum() == 0, f"Invalid data for: {company_name} from {START} to {TODAY}. Please try another ticker or date range."



st.subheader('Stock Price Forecast (Random Forest Regressor)')
st.write('The stock price predictor uses the Random Forest Regressor algorithm which analyses historical price patterns to forecast the future performance of a stock. Please note that future prices are influenced by many other factors and this is just a prediction of how the price could look like in the future based on past performance')

# Prepare the data
prepared_data = prepare_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(prepared_data)

# Train the Random Forest Regressor model
model = train_model(X_train, y_train)

# Prepare future data for prediction
period = st.number_input('Enter the number of days you want to forecast (max. 50 days):', min_value=1, max_value=50, value=10)
future_dates, lagged_features = prepare_future_data(prepared_data, period)

# Make predictions for the future dates
forecast = make_predictions(model, future_dates, lagged_features, period)

# Prepare data for the plot
historical_data = go.Scatter(x=prepared_data.index, y=prepared_data['Close'], mode='lines', name='Historical Close Price')
forecast_data = go.Scatter(x=forecast['Date'], y=forecast['Close'], mode='lines', name='Forecast Close Price')

# Create the plot
fig = go.Figure(data=[forecast_data])

# Set title
fig.update_layout(title= 'Stock Price Forecast')

# Display the plot
st.plotly_chart(fig)




