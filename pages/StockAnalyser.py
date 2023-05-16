#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:07:56 2023

@author: lucas
"""

import streamlit as st
from datetime import date
import re
import yfinance as yf
from plotly import graph_objs as go


st.title("💰🚀 Stock Analyser App 🚀💰")


# Set the minimum and maximum start date values
min_date = date(2010, 1, 1)
max_date = date.today()

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

TODAY = date.today().strftime('%Y-%m-%d')

# Function to assess if the Input for a Ticker is valid 
def is_valid_ticker(ticker):
    #Check if the ticker is valid.
    pattern = r'^[A-Z.]{1,6}$'  # match 1 to 5 uppercase letters
    return re.match(pattern, ticker)

# Check if user_input is valid and return a warning message otherwise 
user_input = st.text_input('Enter Company Name', 'AAPL')
if not is_valid_ticker(user_input):
    st.warning('Please enter a valid stock ticker (e.g. AAPL)')

@st.cache_data
def load_data(ticker, START): 
    data = yf.download(ticker, start=START, end=TODAY, repair=True)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text(f"Loading data for: {user_input} from {START} to {TODAY}...")
data = load_data(user_input, START)
data_load_state.text(f'Loading data for: {user_input} from {START} to {TODAY} is done!')

st.write('------------------------------------------------------------------------')

# Section to Display a summary of the company

# Get Ticker object for the given ticker
ticker_info = yf.Ticker(user_input)

# Get Name of the Stock 
# Retrieve the name of the stock
stock_name = ticker_info.info['longName']

# Fetch company summary
company_summary = ticker_info.info['longBusinessSummary']

# Display the company summary
st.subheader(f'Summary of {stock_name}:')
st.write(company_summary)
st.write('------------------------------------------------------------------------')

# Section to plot the stock performance 

st.subheader(f'Stock Performance of {stock_name}:')
st.write(f'In this section you can see how {stock_name} performed over time. Moreover, you can see the opening and closing price. By moving the slider below the chart you can zoom into specific time frames.')
def plot_raw_data(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text=f'Performance of {stock_name} stock over time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

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


# Retrieve major holders
holders = ticker_info.institutional_holders

# Filter the top 5 major holders
top_holders = holders.head(5)

# Create a bar plot of the major holders
# Retrieve major holders
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
st.write('In this section you can see how the selected stock performed against a Benchmark indice of your choice over a specified time horizon.')
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
