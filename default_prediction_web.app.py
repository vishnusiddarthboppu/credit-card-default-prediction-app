# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:47:14 2026

@author: vishn
"""

import numpy as np
import pickle
import streamlit as st
import os

#loading the saved model 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_loaded_model = pickle.load(open(os.path.join(BASE_DIR, "default_trained_model.sav"), "rb")) 


#creating a function for prediction

def default_prediction(input_data):
     
    #change the input_data into numpy array 
    input_data_as_numpy_array=np.asarray(input_data) 

    #reshape the array as we are p=redicting for one instance/row 
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1) 
    
    prediction=default_loaded_model.predict(input_data_reshaped) 
    print(prediction) 

    if prediction[0]==0: 
        return "The Person is Not a Defaulter"
    else: 
        return "The Person is a Defaulter"
    
    
def main():
    
    #giving a title
    st.title("💳 Credit Card Default Risk Predictor")
    
    #giving a caption
    st.caption("Enter the customer's details to predict whether they will default next month.")
    
    #subheading
    st.subheader("Customer Details")
    
    #getting the input data from the user
    #['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_1', 'pay_2','pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2','bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1','pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
    
    c1, c2 = st.columns(2) #Splits the page into two side-by-side columns, so the form isn’t one long vertical list
    
    limit_bal= c1.number_input("Credit Limit (LIMIT BALANCE)", 10000, 1000000, 50000, step=10000)
    sex=c2.selectbox("Sex", ["F", "M"])
    education=c1.selectbox("Education", ["University", "Graduate school", "High School", "Others", "Unknown", "0"])
    marriage=c2.selectbox("Marital Status", ["Single", "Married", "Other", "0"])
    age=c1.number_input("Age", 21, 79, 30)
    
    
    st.subheader("Repayment Status (-2 to 8)")
    st.caption("-2 = no credit use, -1 = paid in full, 0 = minimum paid, 1-8 = months delayed")
    pay_opts = list(range(-2, 9))
    cols = st.columns(3)
    pay_1 = cols[0].selectbox("PAY_0 (latest month)", pay_opts, index=2)
    pay_2 = cols[1].selectbox("PAY_2", pay_opts, index=2)
    pay_3 = cols[2].selectbox("PAY_3", pay_opts, index=2)
    pay_4 = cols[0].selectbox("PAY_4", pay_opts, index=2)
    pay_5 = cols[1].selectbox("PAY_5", pay_opts, index=2)
    pay_6 = cols[2].selectbox("PAY_6 (oldest month)", pay_opts, index=2)
    
    st.subheader("Bill Amounts")
    st.caption("Credit card statement amount owed in each of the last 6 months (1 = latest, 6 = oldest).Negative values mean the customer overpaid.")
    cols = st.columns(3)
    bill_amt1 = cols[0].number_input("BILL_AMT1 (latest)", value=0)
    bill_amt2 = cols[1].number_input("BILL_AMT2", value=0)
    bill_amt3 = cols[2].number_input("BILL_AMT3", value=0)
    bill_amt4 = cols[0].number_input("BILL_AMT4", value=0)
    bill_amt5 = cols[1].number_input("BILL_AMT5", value=0)
    bill_amt6 = cols[2].number_input("BILL_AMT6 (oldest)", value=0)
    
    
    st.subheader("Payment Amounts")
    st.caption("Amount actually paid in each of the last 6 months (1 = latest, 6 = oldest).Enter 0 if nothing was paid.")
    cols = st.columns(3)
    pay_amt1 = cols[0].number_input("PAY_AMT1 (latest)", min_value=0, value=0)
    pay_amt2 = cols[1].number_input("PAY_AMT2", min_value=0, value=0)
    pay_amt3 = cols[2].number_input("PAY_AMT3", min_value=0, value=0)
    pay_amt4 = cols[0].number_input("PAY_AMT4", min_value=0, value=0)
    pay_amt5 = cols[1].number_input("PAY_AMT5", min_value=0, value=0)
    pay_amt6 = cols[2].number_input("PAY_AMT6 (oldest)", min_value=0, value=0)
    
    
    #code for prediction
    predict = ""
    
    #creating a button for prediction
    if st.button("Predict"):
        predict = default_prediction([limit_bal, sex, education, marriage, age, pay_1, pay_2, pay_3, pay_4, pay_5, pay_6, bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6])
    
    
    st.success(predict)
    
    
    
    
if __name__== "__main__":
    main()
