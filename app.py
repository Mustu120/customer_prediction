import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model pipeline
try:
    pipeline = joblib.load('customer_status_predictor.joblib')
except FileNotFoundError:
    st.error("Model file 'customer_status_predictor.joblib' not found. "
             "Make sure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the feature lists manually based on our training script
# This ensures the DataFrame created has the correct column order
numerical_features = ['Age', 'Number of Dependents', 'Number of Referrals', 'Tenure in Months', 
                    'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 
                    'Monthly Charge', 'Total Charges', 'Total Refunds', 
                    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']
    
categorical_features = ['Gender', 'Married', 'Offer', 'Phone Service', 'Multiple Lines', 
                    'Internet Service', 'Internet Type', 'Online Security', 'Online Backup', 
                    'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 
                    'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract', 
                    'Paperless Billing', 'Payment Method']

all_features_in_order = categorical_features + numerical_features


# --- App UI ---
st.title('Telecom Customer Status Predictor 📈')
st.write("Use the sidebar to input customer details and predict their status: **Stayed, Churned, or Joined**.")

st.sidebar.header('Customer Details')

# Create a dictionary to hold user inputs
inputs = {}

# --- Create UI Elements Dynamically ---

# Numerical features - using sliders
st.sidebar.subheader('Customer Info & Tenure')
inputs['Age'] = st.sidebar.slider('Age', 18, 80, 46)
inputs['Tenure in Months'] = st.sidebar.slider('Tenure in Months', 1, 72, 32)
inputs['Married'] = st.sidebar.selectbox('Married', ['No', 'Yes'])
inputs['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
inputs['Number of Dependents'] = st.sidebar.slider('Number of Dependents', 0, 9, 0)
inputs['Number of Referrals'] = st.sidebar.slider('Number of Referrals', 0, 11, 2)
inputs['Offer'] = st.sidebar.selectbox('Offer', ['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'])

st.sidebar.subheader('Phone & Internet Service')
inputs['Phone Service'] = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
inputs['Multiple Lines'] = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No Phone Service'])
inputs['Internet Service'] = st.sidebar.selectbox('Internet Service', ['Yes', 'No'])
inputs['Internet Type'] = st.sidebar.selectbox('Internet Type', ['Fiber Optic', 'DSL', 'Cable', 'No Internet'])

st.sidebar.subheader('Monthly Usage')
inputs['Avg Monthly Long Distance Charges'] = st.sidebar.slider('Avg Monthly Long Distance Charges', 0.0, 50.0, 22.9)
inputs['Avg Monthly GB Download'] = st.sidebar.slider('Avg Monthly GB Download', 0.0, 85.0, 20.0)

st.sidebar.subheader('Add-on Services')
inputs['Online Security'] = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No Internet Service'])
inputs['Online Backup'] = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No Internet Service'])
inputs['Device Protection Plan'] = st.sidebar.selectbox('Device Protection Plan', ['No', 'Yes', 'No Internet Service'])
inputs['Premium Tech Support'] = st.sidebar.selectbox('Premium Tech Support', ['No', 'Yes', 'No Internet Service'])
inputs['Streaming TV'] = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No Internet Service'])
inputs['Streaming Movies'] = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No Internet Service'])
inputs['Streaming Music'] = st.sidebar.selectbox('Streaming Music', ['No', 'Yes', 'No Internet Service'])
inputs['Unlimited Data'] = st.sidebar.selectbox('Unlimited Data', ['Yes', 'No', 'No Internet Service'])

st.sidebar.subheader('Contract & Billing')
inputs['Contract'] = st.sidebar.selectbox('Contract', ['Month-to-Month', 'One Year', 'Two Year'])
inputs['Paperless Billing'] = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
inputs['Payment Method'] = st.sidebar.selectbox('Payment Method', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])

# These features are part of the model but are calculated from other fields.
# For a live predictor, we use reasonable defaults or sliders.
st.sidebar.subheader('Revenue (Estimated)')
inputs['Monthly Charge'] = st.sidebar.slider('Monthly Charge', 0.0, 120.0, 64.0)
inputs['Total Charges'] = st.sidebar.slider('Total Charges', 0.0, 8700.0, 2283.0)
inputs['Total Refunds'] = st.sidebar.slider('Total Refunds', 0.0, 50.0, 0.0)
inputs['Total Extra Data Charges'] = st.sidebar.slider('Total Extra Data Charges', 0.0, 150.0, 0.0)
inputs['Total Long Distance Charges'] = st.sidebar.slider('Total Long Distance Charges', 0.0, 3600.0, 750.0)
inputs['Total Revenue'] = st.sidebar.slider('Total Revenue', 0.0, 12000.0, 3034.0)


# --- Prediction Logic ---

# Create a DataFrame from the inputs
# Re-order inputs to match the model's expectation
ordered_inputs = {col: inputs[col] for col in all_features_in_order}
input_df = pd.DataFrame([ordered_inputs])


# Predict button
if st.button('Predict Customer Status'):
    try:
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        
        # Get probabilities
        probabilities = pipeline.predict_proba(input_df)
        
        # Get the classes from the model
        classes = pipeline.classes_
        
        # Format for display
        probs_df = pd.DataFrame(probabilities, columns=classes)
        
        # Display result
        st.subheader('Prediction Result')
        if prediction == 'Stayed':
            st.success(f'Predicted Status: **{prediction}**')
        elif prediction == 'Churned':
            st.error(f' Predicted Status: **{prediction}**')
        else:
            st.info(f' Predicted Status: **{prediction}**')

        st.subheader('Prediction Probabilities')
        # Format the probabilities as percentages
        st.dataframe(probs_df.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Input Data Sent to Model:")
        st.dataframe(input_df)