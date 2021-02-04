import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import sklearn



import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px # plotly express
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import os
# Import label encoder 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
import random


import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('randomForest.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(lockdown_types, new_cases, statTwoweeksago, r_naught ):   
    
    if lockdown_types == "Malaysia No Lockdown":
        lockdown_types = 0
        location = 0
    elif lockdown_types == "PKP":
        lockdown_types = 1
        location = 0
    elif lockdown_types == "PKPB":
        lockdown_types = 2
        location = 0
    elif lockdown_types == "PKPP":
        lockdown_types = 3
        location = 0
    elif lockdown_types == "Singapore No Lockdown":
        lockdown_types = 4
        location = 1
    elif lockdown_types == "Singapore Prelude":
        lockdown_types = 8
        location = 1
    elif lockdown_types == "Singapore Circuit Breaker":
        lockdown_types = 5
        location = 1
    elif lockdown_types == "Singapore Phase 1":
        lockdown_types = 6
        location = 1
    elif lockdown_types == "Singapore Phase 2":
        lockdown_types = 7
        location = 1
    elif lockdown_types == "Thailand Pre No Lockdown":
        lockdown_types = 10
        location = 2
    elif lockdown_types == "Thailand Shutdown":
        lockdown_types = 11
        location = 2
    else :
        lockdown_types = 9
        location = 2
     
    # if r_naught > 1 :
        # status = 0
    # else :
        # status = 1
    # Making predictions 
    prediction = classifier.predict( 
        [[lockdown_types, new_cases, statTwoweeksago, r_naught]])
     
    if prediction == 0:
        pred = 'Not Effective'
    else:
        pred = 'Effective'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:gray;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Lockdown Effectiveness Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    lockdown_types = st.selectbox('Lockdown Types',("PKP","PKPB","PKPP","Singapore Prelude","Singapore Circuit Breaker","Singapore Phase 1","Singapore Phase 2","Thailand Shutdown","Malaysia No Lockdown","Singapore No Lockdown","Thailand Pre No Lockdown","Thailand Post No Lockdown"))
    new_cases = st.number_input('Today Cases', min_value=0, max_value=1500)
    r_naught = st.slider('r_naught', float(0), float(15), float(0.20), format="%.2f")
    statTwoweeksago = st.number_input('Two Weeks Before Cases', min_value=0, max_value=1500)
    
    # if previous_case == 0 :
        # r_naught = 0
    # else :
        # r_naught = round(new_cases/previous_case,2)
    
    # if r_naught > 1 :
        # status = 1
    # else :
        # status = 0
    
    result =""
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(lockdown_types, new_cases, statTwoweeksago, r_naught)
        st.success('Lockdown is {} '.format(result))
        
        
     
if __name__=='__main__': 
    main()

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    LockDownType = st.selectbox('Lock down Type', ("PKP", "PKPB","PKPP"))
    CaseNumber = st.number_input("number of cases")
    DateRange = st.number_input("number of date enforced")
    SocialDistancing = st.number_input("Social Distance rate")
    PeriodEnforce = st.number_input("Period enforced")


    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        # prediction method calling
        result = predict(LockDownType, CaseNumber, DateRange, SocialDistancing, PeriodEnforce)
        st.success('The MCO status: {}'.format(result))


if __name__=='__main__':
    main()
