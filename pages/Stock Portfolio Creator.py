#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:08:07 2023

@author: philippkarlschnell
"""

import streamlit as st
import yfinance as yf
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from datetime import datetime
from io import BytesIO


def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100
	fig = px.line(daily_cum_returns, title=title)
	return fig
	
st.set_page_config(page_title = "Stock Portfolio Optimizer", layout = "wide")
st.header("Stock Portfolio Optimizer")

col1, col2 = st.columns(2)

import streamlit as st
import datetime


# Set the start date to January 1, 2013
start_date = datetime.date(2013, 1, 1)

# Set the end date to today's date
end_date = datetime.date.today()

# Create a slider to select the date range
selected_range = st.slider("Select a date range", min_value=start_date, max_value=end_date, value=start_date)

# Display the selected date range
st.write("You selected the following date range:", selected_range, "to", end_date)

# Convert start_date and end_date to string format for Yahoo Finance API
start_date = selected_range.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas, e.g. "MA,FB,V,AMZN,JPM,BA"', '').upper()
tickers = tickers_string.split(',')

# Fetch stock prices for each ticker symbol and store them in a dictionary
stocks_dict = {}
for ticker in tickers:
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        stocks_dict[ticker] = stock['Adj Close']
    except:
        st.write(f"Could not fetch data for {ticker}")

# Combine the stock prices into a single DataFrame
if stocks_dict:
    stocks_df = pd.concat(stocks_dict.values(), axis=1, keys=stocks_dict.keys())
    st.write(stocks_df.head())
else:
    st.write("No data available for the selected tickers.")


stocks_df = None
if stocks_dict:
    stocks_df = pd.concat(stocks_dict.values(), axis=1, keys=stocks_dict.keys())
    st.write(stocks_df.tail())

if stocks_df is not None:
    # Plot Individual Cumulative Returns
    fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
 
    # Get weights
    weights = {}
    for ticker in tickers:
        weight = st.number_input(f"Enter weight for {ticker}", min_value=0.0, max_value=1.0, step=0.01)
        weights[ticker] = weight
    total_weight = sum(weights.values())
    if total_weight != 0:
        weights = {k: v / total_weight for k, v in weights.items()} # normalize weights
    else:
        st.write("Sum of weights is zero. Please enter non-zero weights.")

    
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df.columns = ['weights']

    # Calculate returns of portfolio with weights
    stocks_df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
        stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight

    # Plot Cumulative Returns of Portfolio
    fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Portfolio Starting with $100')

    # Display everything on Streamlit
    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))    
    st.plotly_chart(fig_cum_returns_optimized)

    st.subheader("Portfolio Weights")
    st.dataframe(weights_df)

    # Calculate expected annual return and volatility
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)

    # Display expected annual return and volatility
    st.subheader("Portfolio Performance")
    st.write("Expected annual return: {:.2f}%".format(ef.portfolio_performance()[0]*100))
    st.write("Expected annual volatility: {:.2f}%".format(ef.portfolio_performance()[1]*100))

    st.plotly_chart(fig_cum_returns)
else:
    st.write("No data available for the selected stocks")

