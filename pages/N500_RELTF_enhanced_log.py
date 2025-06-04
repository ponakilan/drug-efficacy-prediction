from Models.Enhanced_Log.model import mlr_base_results,svr_base_results,load_and_prepare_data
import streamlit as st
import pandas as pd
st.title("Enhanced Log Multi Linear Regression and Support Vector Regression ")



input_sets = {
    "Input 1": {"Sex": 2, "Age": 55, "Tumor_Grade": 2,"Tumor_Stage":3,
                "Clinical_Staging":1,"ABCB1":0.088,"ABCC1":0.044,"ABCC2":-0.622,"ABCC3":23.18,"ABCC5":0.117,
                "ABCG2":0.877,"CDK2":0.306,"CDKN1A":-0.0767,"LRP1":0.34,"STAT5B":1.52,"TP53":0.00891},
    "Input 2": {"Sex": 1, "Age": 60, "Tumor_Grade": 3,"Tumor_Stage":1,
                "Clinical_Staging":2,"ABCB1":0.151950,"ABCC1":1.974745,"ABCC2":0.249855	,"ABCC3":0.113834,"ABCC5":8.291693,
                "ABCG2":1.063,"CDK2":1.196594,"CDKN1A":-0.177106,"LRP1":0.34,"STAT5B":0.353145,"TP53":0.002501},
    "Input 3": {"Sex": 1, "Age": 42, "Tumor_Grade": 3,"Tumor_Stage":4,
                "Clinical_Staging":4,"ABCB1":0.029,"ABCC1":0.016,"ABCC2":-1.366,"ABCC3":0.9106,"ABCC5":0.401,
                "ABCG2":1.637,"CDK2":0.455,"CDKN1A":-0.161,"LRP1":0.418,"STAT5B":6.77,"TP53":0.00825}
}

selected_set_name = st.selectbox("Choose a predefined input set:", list(input_sets.keys()))
selected_input = input_sets[selected_set_name]
input_df = pd.DataFrame([selected_input])

# Example usage from Streamlit or another main script
from Models.Enhanced_Log.model import load_and_prepare_data

X,X_scaled, y, label, scaler = load_and_prepare_data("RELTF")  # or "30", or "RELTF"


if st.button("Predict with MLR"):
    output_df = mlr_base_results(input_df,X_scaled,y,label)
    #print(output_df)
    st.success(f"MLR Prediction: success")
    st.table(10**output_df)

if st.button("Predict with SVR"):
    output_df = svr_base_results(input_df,X_scaled,y,label,scaler)
    #print(output_df)
    st.success(f"SVR Prediction: success")
    st.table(10**output_df)
