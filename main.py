import streamlit as st
import pandas as pd
import os
bp = os.getcwd()
path = os.path.join(bp,"dataset/N30ALL.csv")

n30 = pd.read_csv(path)
n500gan = pd.read_csv("dataset/N500_GAN.csv")
n500rel = pd.read_csv("dataset/N_500_RELTF.csv")


pages = {
    "Enhanced Log MLR and SVR":[
        st.Page("pages/N30_SIM_enhanced_log.py",title="N30 dataset"),
        st.Page("pages/N500_GAN_enhanced_log.py",title="GAN dataset"),
        st.Page("pages/N500_RELTF_enhanced_log.py",title="RELTF dataset"),
    ],
    "Residual model":[
        st.Page("pages/N30_SIM_residual.py",title="N30 dataset"),
        st.Page("pages/N500_GAN_residual.py",title="GAN dataset"),
        st.Page("pages/N500_RELTF_residual.py",title="RELTF dataset"),
    ],
    "LightGBM and Multitask NN":[
        st.Page("pages/N30_SIM_LGBM_MNN.py",title="N30 dataset"),
        st.Page("pages/N500_GAN_LGBM_MNN.py",title="GAN dataset"),
        st.Page("pages/N500_RELTF_LGBM_MNN.py",title="RELTF dataset")
    ],
    "Logistic Regression+KNN":[
        st.Page("pages/Ensemble.py",title="RELTF dataset"),
    ]

}
st.navigation(pages).run()
st.sidebar.header("Download Datasets")

with st.sidebar:
    st.download_button("Download N30 Dataset", n30.to_csv(index=False), "N30ALL.csv", "text/csv")
    st.download_button("Download N500 GAN Dataset", n500gan.to_csv(index=False), "N500GAN.csv", "text/csv")
    st.download_button("Download N500 RELTF Dataset", n500rel.to_csv(index=False), "N500RELTF.csv", "text/csv")



#pg = st.navigation([st.Page("pages/n30_1.py")])
#pg.run()









