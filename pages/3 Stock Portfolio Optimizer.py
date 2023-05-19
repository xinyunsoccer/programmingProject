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
from datetime import date, timedelta
from io import BytesIO


# Set the title for the Streamlit app 
st.set_page_config(page_title = "⚙️📈 Stock Portfolio Optimizer 📈⚙️", layout = "wide")
st.header("⚙️📈 Stock Portfolio Optimizer 📈⚙️")

# Create two columns for layout purposes
col1, col2 = st.columns(2)

# Create two columns for layout purposes
col1, col2 = st.columns(2)


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

# Prompt the user to enter stock tickers separated by commas and convert them to uppercase
tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas, e.g. "MSFT, GOOG, META, AAPL."', '').upper()

# Split the tickers string using commas to create a list of tickers
tickers = tickers_string.split(',')


# Fetch stock prices for each ticker symbol and store them in a dictionary
stocks_dict = {}
for ticker in tickers:
    try:
        stock = yf.download(ticker, start=START, end=TODAY, repair=True)
        stocks_dict[ticker] = stock['Adj Close']
    except:
        st.write(f"Could not fetch data for {ticker}")
        
# Display the sentence indicating the number of stocks and the list of stocks in the portfolio
if stocks_dict:
    num_stocks = len(stocks_dict)
    stock_list = ', '.join(tickers)
    st.subheader("Your Portfolio consists of {} Stocks: {}. Each Stock Price in the selected Timeframe is shown below".format(num_stocks, stock_list))  

# Combine the stock prices into a single DataFrame
if stocks_dict:
    stocks_df = pd.concat(stocks_dict.values(), axis=1, keys=stocks_dict.keys())
    st.write(stocks_df.head())
else:
    st.write("No data available for the selected tickers.")


stocks_df = None
# Check if stocks_dict is not empty
if stocks_dict:
    # Concatenate the values in stocks_dict into a single DataFrame
    stocks_df = pd.concat(stocks_dict.values(), axis=1, keys=stocks_dict.keys())
    # Display the last few rows of the DataFrame
    st.write(stocks_df.tail())

# Check if stocks_df is not None
if stocks_df is not None:
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Calculate cumulative returns, convert to daily percentage change, calculate cumulative product, and create a line plot
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
    
    # Calculate expected returns and sample covariance matrix for portfolio optimization later
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)
    
    # Get optimized weights
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.02)
    weights = ef.clean_weights()
    
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df.columns = ['weights']

    # Calculate returns of portfolio with optimized weights
    stocks_df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
        stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
    
    # Plot cumulative returns of optimied portfolio 
    st.subheader('Cumulative Returns of Optimized Portfolio')
    st.write('In this section you can see how your optimized portfolio, consisting of {} stocks, performes. By looking at the cumulative portfolio returns starting with $100.'.format(tickers_string))
    fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
    st.plotly_chart(fig_cum_returns_optimized)
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display optimized weights
    st.subheader("Optimized Max Sharpe Portfolio Weights")
    st.write("The table in this section shows you the optimized weights for each stock {} in your portfolio.".format(tickers_string))
    st.dataframe(weights_df)
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Plot efficient frontier curve
    st.subheader("Optimized Max Sharpe Portfolio Performance")
    st.write("This section displays a plot of the efficient frontier curve. The curve represents the set of optimal portfolios that offer the highest expected return for each level of risk. The star indicates the optimal portfolio that offers the highest expected return while maximizing the Sharpe Ratio.")
    fig = plot_efficient_frontier_and_max_sharpe(mu, S)
    fig_efficient_frontier = BytesIO()
    fig.savefig(fig_efficient_frontier, format="png")
    st.image(fig_efficient_frontier)
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display performance measures 
    st.subheader("Optimized Performance Measures")
    st.write("This section determines the expected annual return, annual volatility and sharpe ratio based on your selected timeframe.")
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
    st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Calculate and plot correlation matrix stocks
    st.subheader("Correlation between Stocks")
    st.write("In this section, you can explore the correlation matrix, which showcases the pairwise correlations among the stocks in your portfolio. In addition, you can compare the correlation between each stock and the optimized portfolio.")
    corr_df = stocks_df.corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
    st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Plot Individual Stock Prices
    st.subheader("Individual Stock Prices")
    st.write("This section illustrates the stock prices of {} compared to your optimized portfolio in the selected timeframe.".format(tickers_string))
    fig_price = px.line(stocks_df, title='Price of Individual Stocks')
    st.plotly_chart(fig_price)
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Plot Individual Cumulative Returns
    st.subheader("Individual Cumulative Returns")
    st.write("This section illustrates the individual cumulative returns, starting with $100, of the stocks {} compared to your optimized portfolio in the selected timeframe.".format(tickers_string))
    fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
    st.plotly_chart(fig_cum_returns)
    
else:
    st.write("No data available for the selected stocks")








