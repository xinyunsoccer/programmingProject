import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
from datetime import date, timedelta


# Set the title for the Streamlit app 
st.set_page_config(page_title = "⚒️📈 Stock Portfolio Creator 📈⚒️", layout = "wide")
st.header("⚒️📈 Stock Portfolio Creator 📈⚒️")

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
tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas, e.g. "MA,FB,V,AMZN,JPM,BA"', '.').upper()

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


# Combine the stock prices into a single DataFrame
st.subheader("Your Portfolio consists of {} Stocks. Each Stock Price in the selected Timeframe is shown below".format(tickers_string))    
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
    
    # Plot Individual Cumulative Returns
    st.subheader('Cumulative Returns of Individual Stocks')
    st.write('In this section you can see how your stocks {} performed over time. By looking at the cumulative returns starting with $100.'.format(tickers_string))
    fig_cum_returns_ind = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
    st.plotly_chart(fig_cum_returns_ind)
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display subheader for manual portfolio adjustment
    st.subheader('Manual Portfolio Adjustment')
    st.write('From now on the portfolio, consisting of {} stocks, can be manually optimized by adjusting the weights of each stock. Keep in mind that the sum of all weights should equal 1.'.format(tickers_string))
    
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
    stocks_df['Manually Adjusted Portfolio'] = 0
    for ticker, weight in weights.items():
        stocks_df['Manually Adjusted Portfolio'] += stocks_df[ticker]*weight

    # Calculate average annual return
    start_date = datetime.strptime(START.strftime('%Y-%m-%d'), '%Y-%m-%d')
    end_date = datetime.strptime(TODAY, '%Y-%m-%d')

    days = (end_date - start_date).days
    years = days / 365.25
    portfolio_cumulative_return = stocks_df['Manually Adjusted Portfolio'].iloc[-1] / stocks_df['Manually Adjusted Portfolio'].iloc[0]
    average_annual_return = (portfolio_cumulative_return ** (1/years)) - 1
    
    # Calculate average annual volatility
    daily_returns = stocks_df['Manually Adjusted Portfolio'].pct_change()
    volatility = daily_returns.std()
    average_annual_volatility = volatility * (252 ** 0.5)  # Assuming 252 trading days in a year
    
    # Calculate standard deviation
    portfolio_std_dev = daily_returns.std()

    # Calculate downside deviation
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std()

    # Calculate maximum drawdown
    portfolio_values = stocks_df['Manually Adjusted Portfolio']
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    # Format the maximum drawdown
    max_drawdown = round(max_drawdown * 100, 2)
    # Format the start and end dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Calculate drawdown duration 
    drawdown_duration = (drawdown == 0).astype(int).groupby((drawdown != 0).cumsum()).sum().max()
    # Format the drawdown duration
    drawdown_duration_str = str(drawdown_duration) + " days"
    
    # Slider to adjust the risk free rate for the sharpe ratio 
    risk_free_rate = st.slider(
        'Adjust realistic Risk-Free Rate for Sharpe Ratio',
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01
        )
    
    # Calculate sharpe ratio 
    portfolio_returns = stocks_df['Manually Adjusted Portfolio'].pct_change()
    portfolio_volatility = portfolio_returns.std() * (252 ** 0.5)  # Assuming 252 trading days in a year

    portfolio_excess_returns = portfolio_returns - risk_free_rate
    sharpe_ratio = portfolio_excess_returns.mean() / portfolio_volatility
    
    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)

    # Display average annual return, volatility, standard deviation, downside deviation, maximum drawdown, drawdown duration, sharpe ratio 
    st.subheader("Past Performance Measures of Portfolio in the selected Timeframe from" + " " + start_date_str + " to " + end_date_str + "")
    st.write("Average Annual Portfolio Return:", round(average_annual_return*100, 2),"%", unsafe_allow_html=True, style={"color": "white"})
    st.write("Average Annual Portfolio Volatility:", round(average_annual_volatility*100, 2), "%", unsafe_allow_html=True, style={"color": "white"})
    st.write("Standard Deviation:", round(portfolio_std_dev * 100, 2), "%", unsafe_allow_html=True, style={"color": "white"})
    st.write("Downside Deviation:", round(downside_deviation * 100, 2), "%", unsafe_allow_html=True, style={"color": "white"})
    st.write("Maximum Drawdown:", max_drawdown, "%", unsafe_allow_html=True, style={"color": "white"})
    st.write("Drawdown Duration:", drawdown_duration_str, unsafe_allow_html=True, style={"color": "white"})
    st.write("Sharpe Ratio:", round(sharpe_ratio, 2), "%", unsafe_allow_html=True, style={"color": "white"})

    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display subheader for cumulative returns of portfolio
    st.subheader('Cumulative Returns of Portfolio')
    st.write('In this section you can see how your portfolio, consisting of {} stocks, performed over time. By looking at the cumulative portfolio returns starting with $100.'.format(tickers_string))
    
    # Plot Cumulative Returns of Portfolio
    fig_cum_returns = plot_cum_returns(stocks_df['Manually Adjusted Portfolio'], 'Cumulative Returns of Portfolio Starting with $100')
    # Display the plot of cumulative returns
    st.plotly_chart(fig_cum_returns)

else:
    st.write("No data available for the selected stocks.")


# Add a line to separate subheaders
st.markdown("<hr>", unsafe_allow_html=True)


# Display explanations for better understanding the past performance measures 
st.subheader("Explanation of Performance Measures")
st.write("Use this section if you don't fully understand the Performance Measures which have been used to determine the performance of your portfolio.")

# Add explanation for average annual portfolio return 
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Average Annual Portfolio Return")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button1 = button_col1.button("Explanation of Average Annual Portfolio Return", key="info_button1")

# Display information or the initial layout based on button click
if info_button1:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("Average annual portfolio return calculates the average percentage gain or loss of a portfolio over a year, representing the annualized performance of the investment. It provides a measure of the portfolio's average profitability or loss on an annual basis, helping investors evaluate the performance and potential returns of their investment.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Average Annual Portfolio Return")


# Add empty subheader for seperation
st.subheader('')


# Add explanation for average annual portfolio volatility
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Average Annual Portfolio Volatility")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button2 = button_col1.button("Explanation of Average Annual Portfolio Volatility", key="info_button2")

# Display information or the initial layout based on button click
if info_button2:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("Average annual portfolio volatility measures the fluctuation or variability of a portfolio's returns over a year. It quantifies the degree of risk associated with the portfolio and provides insights into its potential price movements and stability.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Average Annual Portfolio Volatility")


# Add empty subheader for seperation
st.subheader('')


# Add explanation for standard deviation
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Standard Deviation")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button3 = button_col1.button("Explanation of Standard Deviation", key="info_button3")

# Display information or the initial layout based on button click
if info_button3:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("Standard deviation in a portfolio measures the dispersion or variability of the portfolio's returns from its average return. It provides a measure of the portfolio's risk by indicating how much the individual returns deviate from the average, helping investors assess the potential volatility of their investment.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Standard Deviation")


# Add empty subheader for seperation
st.subheader('')


# Add explanation for downside deviation
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Downside Deviation")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button4 = button_col1.button("Explanation of Downside Deviation", key="info_button4")

# Display information or the initial layout based on button click
if info_button4:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("Downside deviation in a portfolio measures the dispersion or variability of the portfolio's negative returns from its average negative return. It focuses specifically on downside risk by quantifying the extent to which the portfolio's returns deviate below the average, providing investors with insight into the potential downside volatility of their investment.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Downside Deviation")


# Add empty subheader for seperation
st.subheader('')

# Add explanation for maximum drawdown 
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Maximum Drawdown")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button5 = button_col1.button("Explanation of Maximum Drawdown", key="info_button5")

# Display information or the initial layout based on button click
if info_button5:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("Maximum drawdown measures the largest percentage decline from a portfolio's peak value to its lowest point. It gives an idea of the portfolio's risk and potential losses.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Maximum Drawdown")


# Add empty subheader for seperation
st.subheader('')


# Add explanation for drawdown duration
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Drawdown Duration")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button6 = button_col1.button("Explanation of Drawdown Duration", key="info_button6")

# Display information or the initial layout based on button click
if info_button6:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("Drawdown duration measures the time it takes for an investment or portfolio to recover from a decline and reach its previous peak value. It provides insight into how long an investment remains below its previous high and helps assess its recovery period.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Drawdown Duration")


# Add empty subheader for seperation
st.subheader('')


# Add explanation for sharpe ratio 
# Define the initial layout
show_info = False
close_info = False
# Display the heading and info button
drawdown_header = st.empty()
drawdown_header.write("Sharpe Ratio")

# Create a column layout for buttons
button_col1, button_col2 = st.columns(2)
info_button7 = button_col1.button("Explanation of Sharpe Ratio", key="info_button7")

# Display information or the initial layout based on button click
if info_button7:
    show_info = True

if show_info:
    # Display the information text
    drawdown_header.empty()
    st.write("The Sharpe Ratio is a measure of risk-adjusted return that quantifies the excess return earned per unit of risk taken by an investment or portfolio. It compares the return of the investment above a risk-free rate to the volatility of the investment, providing investors with a metric to evaluate the efficiency of an investment in generating returns relative to its risk.")
    close_info = button_col2.button("Close", key="close_info")
    if close_info:
        show_info = False

# Display the initial layout when info is not shown
if not show_info and not close_info:
    drawdown_header.write("Sharpe Ratio")

