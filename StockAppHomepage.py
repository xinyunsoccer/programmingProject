#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:24:33 2023

@author: lucas
"""
# import streamlit library 
import streamlit as st 

# set up the page configuration 
st.set_page_config(
    page_title='🚀📈 Stock Analytics and Portfolio App 📈🚀',
    page_icon="📈", 
    initial_sidebar_state='expanded', 
    layout = 'wide'
    )

# set the Title for our App
st.title(' 🚀📈 Stock Analytics and Portfolio App 📈🚀')

# add a Description for the App 
st.subheader('Welcome to our Programming Project Page') 
st.write("""We have created a page with which you can make more informed investment decisions. (1) First of all, you can analyse the past performance and KPIs of a stock of your choice and predict its future price.
 (2) On the second page you can create a portfolio of your choice based on the stocks you have analysed before and manually decide how much you weigh each stock. Based on this you can see the performance of your created portfolio. 
 (3) Lastly, you can go to the Stock Portfolio Optimizer where you can input the stocks you picked for your portfolio and check how the implemented portfolio Optimizer would distribute the weights of your selected portfolio. Have fun!""")

# Display the indicator to select a page
st.sidebar.success('Select a page above.')
