#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from predict_page import show_predict_page
from Test import build_model
build_model()
show_predict_page()

