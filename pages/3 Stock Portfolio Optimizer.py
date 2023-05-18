# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import streamlit as st
import yfinance as yf
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from datetime import date
from io import BytesIO


def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100
	fig = px.line(daily_cum_returns, title=title)
	return fig
	
def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.deepcopy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

st.set_page_config(page_title = "⚙️📈 Stock Portfolio Optimizer 📈⚙️", layout = "wide")
st.header("⚙️📈 Stock Portfolio Optimizer 📈⚙️")

col1, col2 = st.columns(2)

# Set the start date to January 1, 2010
start_date = date(2010, 1, 1)

# Set the end date to today's date
end_date = date.today()

# Create a slider to select the date range
selected_range = st.slider("Select a date range", min_value=start_date, max_value=end_date, value=date(2015, 1, 1))

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
    # Plot Individual Stock Prices
    fig_price = px.line(stocks_df, title='Price of Individual Stocks')
    # Plot Individual Cumulative Returns
    fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
    # Calculatge and Plot Correlation Matrix between Stocks
    corr_df = stocks_df.corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')

    # Calculate expected returns and sample covariance matrix for portfolio optimization later
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)

    # Plot efficient frontier curve
    fig = plot_efficient_frontier_and_max_sharpe(mu, S)
    fig_efficient_frontier = BytesIO()
    fig.savefig(fig_efficient_frontier, format="png")

    # Get optimized weights
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.02)
    weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df.columns = ['weights']

    # Calculate returns of portfolio with optimized weights
    stocks_df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
        stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight

    # Plot Cumulative Returns of Optimized Portfolio
    fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')

    # Display everything on Streamlit
    st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))    
    st.plotly_chart(fig_cum_returns_optimized)

    st.subheader("Optimized Max Sharpe Portfolio Weights")
    st.dataframe(weights_df)

    st.subheader("Optimized Max Sharpe Portfolio Performance")
    st.image(fig_efficient_frontier)

    st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
    st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))

    st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
    st.plotly_chart(fig_price)
    st.plotly_chart(fig_cum_returns)
else:
    st.write("No data available for the selected stocks")








