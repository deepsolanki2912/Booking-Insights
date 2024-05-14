import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the DataFrame
bookings = pd.read_csv('hotel_bookings.csv')

# Load the model
pipe = pickle.load(open('pipe1.pkl', 'rb'))

st.title("Hotel Bookings")

hotel_types = bookings['hotel'].unique()
room_types = bookings['reserved_room_type'].unique()
countries = bookings['country'].unique()
arrival_years = bookings['arrival_date_year'].unique()
arrival_months = bookings['arrival_date_month'].unique()

hotel = st.selectbox('Hotel Type:', hotel_types)
adults = st.number_input("Adults:", min_value=0, max_value=10, value=0)
children = st.number_input("Children:", min_value=0, max_value=10, value=0)
country = st.selectbox('Country (Destination):', countries)
arrival_date_year = st.selectbox("Year:", arrival_years)
arrival_date_month = st.selectbox('Month:', arrival_months)
avg_daily_rate = st.number_input("Average Daily Rate:", min_value=40, value=40)
assigned_room_type = st.selectbox('Assigned room type', room_types)
reserved_room_type = st.selectbox('Reserved Room Type:', room_types)

le=LabelEncoder()

def prd(p):
    if hotel == 'Resort Hotel':
        
        if p == 0:
            return "There are less chances of cancelation"
        else:
            return "There are more chances of cancelation"

    else:
        if p == 0:
            return "There are more chances of cancelation"
        else:
            return "There are less chances of cancelation"

if st.button('Predict Cancelation'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'hotel': [hotel],
        'country': [country],
        'arrival_date_year': [arrival_date_year],
        'arrival_date_month': [arrival_date_month],
        'assigned_room_type': [assigned_room_type],
        'reserved_room_type': [reserved_room_type],
        'adr': [avg_daily_rate],
        'children': [children],
        'adults': [adults]
    })
    
    # Apply label encoding
    list_1=list(input_data.columns)
    list_cate=[]
    for i in list_1:
       if input_data[i].dtype=='object':
           list_cate.append(i)
           
    for i in list_cate:
       input_data[i]=le.fit_transform(input_data[i])

    query = input_data.values.reshape(1, -1)

    p = pipe.predict(query)
    st.write(prd(p))
    # index = list_1.index(hotel)
    # if (input_data['hotel'] == 0).any():
    #     p = pipe.predict(query)
    #     st.write(prd(p))
    # q = pipe.predict(query)
    # st.write(prd(q))
