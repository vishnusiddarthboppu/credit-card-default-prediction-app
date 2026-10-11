# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pickle 

#loading the saved model 

default_loaded_model=pickle.load(open("C:/Users/vishn/OneDrive/Attachments/credit_card_Default_project/default_trained_model.sav","rb")) 

input_data=(50000,"M","University","Married",44,0,0,0,0,0,0,45335,46027,30286,26275,26823,27371,1524,1427,941,972,992,1000 ) 

#change the input_data into numpy array 
input_data_as_numpy_array=np.asarray(input_data) 

#reshape the array as we are p=redicting for one instance/row 
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1) 
prediction=default_loaded_model.predict(input_data_reshaped) 
print(prediction) 

if prediction[0]==0: 
    print("The Person is Not a Defaulter") 
else: 
    print("The Person is a Defaulter") 
    