#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:24:33 2023

@author: lucas
"""

import streamlit as st 

st.set_page_config(
    page_title='Stock Analytics App',
    page_icon="📈", 
    initial_sidebar_state='expanded'
    )

st.title(' 📈 Stock Analytics App 📈')

#Create a more in-depth description later

st.subheader('Welcome to our Programming Project Page') 
st.write("""We have created several sub functions with which you can explore stock price data, analyse it and predict future prices.\n
To access the respective pages navigate in the menu on the left. Have fun!""")

st.sidebar.success('Select a page above.')
