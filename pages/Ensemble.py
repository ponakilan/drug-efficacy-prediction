from Models.Enhanced_Log.ensemble import predict_drug
import streamlit as st
import pandas as pd
st.title("LOGISTIC REGRESSION AND KNN")

st.subheader("N500 RELTF Dataset")


input_sets = {
    "Input 1": {"Sex": 2, "Age": 55, "Tumor_Grade": 2,"Tumor_Stage":3,
                "Clinical_Staging":1,"ABCB1":0.088,"ABCC1":0.044,"ABCC2":-0.622,"ABCC3":23.18,"ABCC5":0.117,
                "ABCG2":0.877,"CDK2":0.306,"CDKN1A":-0.0767,"LRP1":0.34,"STAT5B":1.52,"TP53":0.00891},
    "Input 2": {"Sex": 1, "Age": 48, "Tumor_Grade": 2,"Tumor_Stage":1,
                "Clinical_Staging":2,"ABCB1":0.00172,"ABCC1":0.00669,"ABCC2":-2.120	,"ABCC3":0.011,"ABCC5":0.00322,
                "ABCG2":0.00297,"CDK2":-2.196,"CDKN1A":-2.458,"LRP1":0.0110,"STAT5B":0.0158,"TP53":1.0328},
    "Input 3": {"Sex": 1, "Age": 42, "Tumor_Grade": 3,"Tumor_Stage":4,
                "Clinical_Staging":4,"ABCB1":0.029,"ABCC1":0.016,"ABCC2":-1.366,"ABCC3":0.9106,"ABCC5":0.401,
                "ABCG2":1.637,"CDK2":0.455,"CDKN1A":-0.161,"LRP1":0.418,"STAT5B":6.77,"TP53":0.00825}
}


selected_set_name = st.selectbox("Choose a predefined input set:", list(input_sets.keys()))
selected_input = input_sets[selected_set_name]
input_df = pd.DataFrame([selected_input])





if st.button("Predict"):
    output = predict_drug(input_df)
    #print(output_df)
    st.success(f"Prediction: success")
    st.write("Predicted drug is:",output)


