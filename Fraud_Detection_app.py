import streamlit as st
import pandas as pd
from joblib import load

# Loading the model
model = load('fraud_detection_model.joblib')

# Creating the streamlit app
st.title('Fraud Detection App')

# Cretaing Input fields
type = st.selectbox("Type of Transaction", ('CASH IN', 'CASH OUT', 'DEBIT', 'TRANSFER', 'PAYMENT'))
amount = st.number_input('Enter the transaction amount', min_value=0)
oldbalorig = st.number_input('Enter Balance of the Origin Customer before transaction (in Rs.)', min_value=0)
newbalorig = st.number_input('Enter Balance of the Origin Customer after transaction (in Rs.)', min_value=0)
oldbaldest = st.number_input('Enter Balance of the Destination Customer before transaction (in Rs.)', min_value=0)
newbaldest = st.number_input('Enter Balance of the Destination Customer after transaction (in Rs.)', min_value=0)

# Creating new features
bal_diff_orig = newbalorig - oldbalorig
bal_diff_dest = newbaldest - oldbaldest
bal_diff_orig_amt = bal_diff_orig / amount
bal_diff_dest_amt = bal_diff_dest / amount

# Type one hot encoding encoding
cash_out = 0
debit = 0
transfer = 0
payment = 0

if type == 'CASH OUT':
    cash_out = 1
elif type == 'DEBIT':
    debit = 1
elif type == 'TRANSFER':
    transfer = 1
elif type == 'PAYMENT':
    payment = 1
else:
    pass
# Prediction button
pred_button = st.button('Predict')
# Make a prediction using the model
if pred_button:
    prediction = model.predict([[amount, oldbalorig, newbalorig, oldbaldest, newbaldest, bal_diff_orig, bal_diff_dest, bal_diff_orig_amt, bal_diff_dest_amt, cash_out, debit, payment, transfer]])

    # Display the prediction result on the main screen
    st.header("Prediction Result")
    if prediction[0] == 0:
        st.success("This Transaction is not a Fraud.")
    else:
        st.error("This Transaction is a Fraud.")