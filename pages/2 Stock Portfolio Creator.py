import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        st.write("Could not fetch data.")

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
        st.write("Sum of weights is zero. Please enter non-zero weights which add up to 1.")

    
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
    

    # Add a line to separate subheaders
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display subheader for cumulative returns of portfolio
    st.subheader('Cumulative Returns of Portfolio')
    st.write('In this section you can see how your portfolio, consisting of {} stocks, performed over time. By looking at the cumulative portfolio returns starting with $100.'.format(tickers_string))
    
    # Plot Cumulative Returns of Portfolio
    fig_cum_returns = plot_cum_returns(stocks_df['Manually Adjusted Portfolio'], 'Cumulative Returns of Portfolio Starting with $100')
    # Display the plot of cumulative returns
    st.plotly_chart(fig_cum_returns)
    
    # Header for risk-return tradeoff chart
    st.subheader('Risk-Return Tradeoff of individual Stocks in Portfolio')
    
    # Display the description with bullet point enumeration
    st.markdown('In this section the bubble chart illustrates the tradeoff between the risk and return of your stocks. To use this chart, please adjust the risk free rate.')
    st.markdown('- The x-axis shows the annual return of the stocks.')
    st.markdown('- The y-axis shows the volatility of the stocks.')
    st.markdown('- The size of the element represents the weight of the stock in the portfolio.')
    st.markdown('- The shape of the element represents the Sharpe Ratio. Triangle is < 0.1 | Cross is 0.1 < 0.4 | Circle is 0.4 < 0.8 | Diamond-Star is > 0.8')
    
    # Slider to adjust the risk free rate for the sharpe ratio 
    risk_free_rate = st.slider(
        'Adjust realistic Risk-Free Rate for Sharpe Ratio',
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01, 
        key="risk_free_rate2_slider"
        )
    
    # Calculate risk-adjusted measures
    daily_returns = stocks_df.pct_change()
    annual_returns = daily_returns.mean() * 252  # Assuming 252 trading days in a year
    annual_volatility = daily_returns.std() * (252 ** 0.5)  # Assuming 252 trading days in a year
    sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

    # Create a DataFrame to store the risk-return tradeoff data
    tradeoff_data = pd.DataFrame({
        'Ticker': tickers[:len(stocks_dict)],
        'Annual Return': annual_returns[:len(stocks_dict)],
        'Volatility': annual_volatility[:len(stocks_dict)],
        'Weights': weights_df['weights'].values[:len(stocks_dict)],
        'Sharpe Ratio': sharpe_ratio[:len(stocks_dict)]
    })

    # Filter the DataFrame to include only tickers with available data
    tradeoff_data = tradeoff_data[tradeoff_data['Ticker'].isin(stocks_dict.keys())]
    
    # Check if there are any tickers with missing data
    missing_tickers = set(tickers) - set(tradeoff_data['Ticker'])

    # Display the tickers with missing data
    if missing_tickers:
        st.write('The following tickers have missing data:')
        st.write(missing_tickers)
    else:
        # Assign different shapes to Sharpe ratio ranges
        def assign_shape(sharpe):
            if sharpe < 0.1:
                return 'triangle-down'
            elif 0.1 <= sharpe < 0.4:
                return 'x'
            elif 0.4 <= sharpe < 0.8:
                return 'circle'
            elif 0.8 <= sharpe <= 5.0:
                return 'star-diamond'

        # Map Sharpe ratios to shapes
        tradeoff_data['Shape'] = tradeoff_data['Sharpe Ratio'].apply(assign_shape)

        colors = tradeoff_data['Ticker'].map(lambda ticker: ord(ticker[0]) % 10)  # Map tickers to a color index

        # Create a bubble chart to visualize the risk-return tradeoff
        fig_tradeoff = go.Figure()

        for i, row in tradeoff_data.iterrows():
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=[row['Annual Return']],
                    y=[row['Volatility']],
                    mode='markers',
                    name=row['Ticker'],  # Set the name of the trace to the ticker
                    marker=dict(
                        size=np.sqrt(row['Weights']) * 200,  # Adjust the size of the bubble based on the weight
                        sizemode='diameter',
                        sizeref=0.1,
                        sizemin=5,
                        symbol=[assign_shape(row['Sharpe Ratio'])], # Use different shapes based on the Sharpe ratio
                        line=dict(
                            width=1,
                            color='black'
                        ),
                        color=colors[i],  # Assign different colors based on the ticker index
                        colorscale='Jet',  # Choose a colorscale for the bubbles
                        opacity=0.8,
                    ),
                    hovertemplate='<b>Ticker:</b> ' + row['Ticker'] +
                          '<br><b>Annual Return:</b> %{x}' +
                          '<br><b>Volatility:</b> %{y}' +
                          '<br><b>Weight:</b> ' + str(row['Weights']) +
                          '<br><b>Sharpe Ratio:</b> ' + str(row['Sharpe Ratio']),
                )
            )

    # Set the x-axis and y-axis labels
    fig_tradeoff.update_layout(
        xaxis_title='Annual Return',
        yaxis_title='Volatility',
        title='Risk-Return Tradeoff for the Portfolio',
        showlegend=True,
        height=600,
    )

    # Display the risk-return tradeoff chart
    st.plotly_chart(fig_tradeoff)

else:
    st.write("No data available for the selected stocks.")


# Add a line to separate subheaders
st.markdown("<hr>", unsafe_allow_html=True)

# Display explanations of performance measures for the portfolio
explanations = {
    "Average Annual Portfolio Return": "Average annual portfolio return calculates the average percentage gain or loss of a portfolio over a timeframe, representing the annualized performance of the investment.",
    "Average Annual Portfolio Volatility": "Average annual portfolio volatility measures the fluctuation or variability of a portfolio's returns over a timeframe, quantifying the degree of risk associated with the portfolio.",
    "Standard Deviation": "Standard deviation in a portfolio measures the dispersion or variability of the portfolio's returns from its average return, providing a measure of the portfolio's risk.",
    "Downside Deviation": "Downside deviation in a portfolio measures the dispersion or variability of the portfolio's negative returns from its average negative return, focusing on downside risk.",
    "Maximum Drawdown": "Maximum drawdown measures the largest percentage decline from a portfolio's peak value to its lowest point, giving an idea of the portfolio's risk and potential losses.",
    "Drawdown Duration": "Drawdown duration measures the time it takes for an investment or portfolio to recover from a decline and reach its previous peak value, providing insight into its recovery period.",
    "Sharpe Ratio": "The Sharpe Ratio is a measure of risk-adjusted return that quantifies the excess return earned per unit of risk taken by an investment or portfolio."
}

st.subheader("Explanation of Performance Measures")
st.write("Use this section if you don't fully understand the Performance Measures which have been used to determine the performance of your portfolio.")

# Iterate through each measure and its corresponding explanation
for measure, explanation in explanations.items():
    show_info = False
    close_info = False
    drawdown_header = st.empty()
    drawdown_header.write(measure)

    # Create two columns for the info button and close button
    button_col1, button_col2 = st.columns(2)
    info_button = button_col1.button(f"Explanation of {measure}", key=f"info_button_{measure}")

    if info_button:
        show_info = True

    if show_info:
        # Display the explanation when the info button is clicked
        drawdown_header.empty()
        st.write(explanation)
        close_info = button_col2.button("Close", key="close_info")
        if close_info:
            show_info = False

    if not show_info and not close_info:
        # Display the measure name if the explanation is not shown
        drawdown_header.write(measure)

    # Add a subheader separator after each measure's explanation
    st.subheader('')


