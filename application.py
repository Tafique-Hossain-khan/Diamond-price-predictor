import streamlit as st

from src.pipeline.prediction_pipeline import CustomInput,PredictPipeline

st.header('Dimond price predictor')


carat = st.text_input("Carat", placeholder="Enter the carat weight of the diamond")
dropdown_options = [
    ["Ideal", "Good", "Very Good", "Fair", "Premium"],
    ["G", "D", "E", "F", "H", "I", "J",],
    ["VS1", "VS2", "SI1", "SI2", "I1", "IF","VVS2","VVS1"],
]

# Create a drop down for the cut
cut = st.selectbox("Cut:", dropdown_options[0])

# Create a drop down for the color
color = st.selectbox("Color:", dropdown_options[1])

# Create a drop down for the clarity
clarity = st.selectbox("Clarity:", dropdown_options[2])

depth = st.text_input("depth", placeholder="Enter the  depth of the diamond")
table = st.text_input("table", placeholder="Enter the  table of the diamond")

x = st.text_input("X", placeholder="Enter the  x of the diamond")
y = st.text_input("Y", placeholder="Enter the  y of the diamond")
z = st.text_input("Z", placeholder="Enter the  z of the diamond")

def perform_prediction():
    obj = CustomInput(carat,cut,color,clarity,depth,table,x,y,z)
    featers = obj.get_df()
    predict_obj = PredictPipeline()
    price = predict_obj.predict(featers)
    return price

submit_button = st.button("Submit")

# If the submit button is clicked, call the my_function() function
if submit_button:
    st.write(f'The price of Diamond is:{perform_prediction()}')









