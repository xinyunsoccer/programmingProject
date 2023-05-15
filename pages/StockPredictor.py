#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:07:56 2023

@author: lucas
"""

import streamlit as st
from datetime import date, timedelta

import re
import yfinance as yf 
from prophet import prophet 
from prophet.plot import plot_plotly 
from plotly import graph_objs as go 

st.title("💰🚀 Stock Predictor App 🚀💰")


# Set the minimum and maximum start date values
min_date = date(2010, 1, 1)
max_date = date.today() - timedelta(days=1)

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
    """Check if the ticker is valid."""
    pattern = r'^[A-Z.]{1,6}$'  # match 1 to 5 uppercase letters
    return re.match(pattern, ticker)

# Check if user_input is valid and return a warning message otherwise 
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
if not is_valid_ticker(user_input):
    st.warning('Please enter a valid stock ticker (e.g. AAPL)')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years*365

@st.cache_data
def load_data(ticker, START): 
    data = yf.download(ticker, start=START, end=TODAY, repair=True)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text(f"Loading data for: {user_input} from {START} to {TODAY}...")
data = load_data(user_input, START)
data_load_state.text(f'Loading data for: {user_input} from {START} to {TODAY} is done!')

# Get Ticker object for the given ticker
ticker_info = yf.Ticker(user_input)

# Fetch company summary
company_summary = ticker_info.info['longBusinessSummary']

# Display the company summary
st.subheader('Company Summary:')
st.write(company_summary)

st.subheader('Raw Data')
st.write(data.head())

def plot_raw_data(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecasting 
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(data.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
