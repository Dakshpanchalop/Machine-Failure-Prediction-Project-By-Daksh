# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:40:26 2024

@author: prati
"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Type_L = st.sidebar.selectbox('Type_Low',('0','1'))
    Type_M = st.sidebar.selectbox('Type_Medium',('0','1'))
    Type_H = st.sidebar.selectbox('Type_High',('0','1'))
    Air = st.sidebar.number_input("Air temperature [K]")
    Process = st.sidebar.number_input("Process temperature [K]")
    Rotational = st.sidebar.number_input("Rotational speed [rpm]")
    Torque = st.sidebar.number_input("Torque [Nm]")
    Tool = st.sidebar.number_input("Tool wear [min]")
    TWF = st.sidebar.selectbox('TWF',('0','1'))
    HDF = st.sidebar.selectbox('HDF',('0','1'))
    PWF = st.sidebar.selectbox('PWF',('0','1'))
    OSF = st.sidebar.selectbox('OSF',('0','1'))
    RNF = st.sidebar.selectbox('RNF',('0','1'))

    data = {'Air temperature':Air,
            'Process temperature':Process,
            'Rotational speed':Rotational,
            'Torque':Torque,
            'Tool wear':Tool,
            'TWF':TWF,
            'HDF':HDF,
            'PWF':PWF,
            'OSF':OSF,
            'RNF':RNF,
            'Type_H':Type_H,
            'Type_L':Type_L,
            'Type_M':Type_M}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()

st.subheader('Data Summary')
st.write('- Type_L: 1 if 50% product quality variant else 0')
st.write('- Type_M: 1 if 30% product quality variant else 0')
st.write('- Type_H: 1 if 20% product quality variant else 0')
st.write('- Air temperature [K]: The ambient air temperature around the machine, measured in Kelvin (K). ')
st.write('- Process temperature [K]: The temperature within the machine’s process, also measured in Kelvin (K). ')
st.write('- Rotational speed [rpm]: The speed at which the machine’s components are rotating, measured in revolutions per minute (rpm).')
st.write('- Torque [Nm]: The amount of torque being applied by the machine, measured in Newton-meters (Nm). ')
st.write('- Tool wear [min]: The amount of wear experienced by the machine tool, measured in minutes. ')
st.write('- TWF (Tool Wear Failure): A binary indicator showing if the failure was due to tool wear. ')
st.write('- HDF (Heat Dissipation Failure): A binary indicator showing if the failure was due to issues with heat dissipation. ')
st.write('- PWF (Power Failure): A binary indicator showing if the failure was due to power-related issues. ')
st.write('- OSF (Overstrain Failure): A binary indicator showing if the failure was due to overstrain. ')
st.write('- RNF (Random Failure): A binary indicator showing if the failure was due to random, unforeseen factors. ')

st.subheader('User Input parameters')
st.write(df)

Maintenance = pd.read_csv("C:\\Users\\PC\\Desktop\\ExcelR Project\\Maintenance_new.csv")
Maintenance.drop(["UDI","Product ID"],inplace=True,axis = 1)

# One-hot encode categorical variables
data_new_encoded = pd.get_dummies(Maintenance, columns=['Type'])

X = data_new_encoded.drop(['Machine failure'], axis = 1)
Y = data_new_encoded['Machine failure']

clf = LogisticRegression(solver='lbfgs', max_iter=1000)
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Machine Failure')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)