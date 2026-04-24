import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(layout="wide")


scaler=joblib.load("scale.pkl")
model=joblib.load("mlmodel.pkl")

st.title("Resturant Rating Predictor")

st.caption("This app help you predict resturant rating class")
st.divider()

avgcost=st.number_input("Please Enter the average cost for two:",min_value=50,max_value=999999,value=1000,step=200 )
table_booking=st.selectbox("Resturant has booking ?",["yes","no"])
online_Delivery=st.selectbox("Resturant has  online Delivery ?",["yes","no"])
pricerange=st.selectbox("Enter the price range(1 cheapest,4 Expnsive)",[1,2,3,4 ])

predictbuton=st.button("Predict Review!")
st.divider()



if online_Delivery=='yes':
    online_Delivery_encoded=1
else:
    online_Delivery_encoded=0

if table_booking=='yes':
    table_booking_encoded=1
else:
    table_booking_encoded=0



values = [[avgcost,table_booking_encoded,online_Delivery_encoded, pricerange]]
my_x_values=np.array(values)


if predictbuton:
    st.balloons()
    X = scaler.transform(my_x_values)
    prediction = model.predict(X)[0]
    #st.write(f"Predicted Rating Class: {prediction}")
    if prediction <2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")