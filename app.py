import streamlit as st
import numpy as np
from pandas import DataFrame
from utils import IT_HR_Classifier
import os
import json
import pickle

st.set_page_config(
    page_title="IT-HR Classification Model",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

st.title("IT-HR Classification Model")
st.header("")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   This Model utilizes Sentence Tranformer embeddings and Logistic Regression to classify tickets into IT and HR cases.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste document **")

with open("it_hr_model", "rb") as f:
    model = pickle.load(f)

doc1 = st.text_area("Paste your text below (max 50 words)", height=150)

MAX_WORDS = 50
import re
res = len(re.findall(r"\w+", doc1))
if res > MAX_WORDS:
    st.warning(
        "⚠️ Your text contains "
        + str(res)
        + " words."
        + " Only the first 50 words will be reviewed."
    )

doc1 = doc1[:MAX_WORDS]

doc2 = st.text_area("Paste your text below (max 500 words)", height=350)

MAX_WORDS = 500
import re
res = len(re.findall(r"\w+", doc2))
if res > MAX_WORDS:
    st.warning(
        "⚠️ Your text contains "
        + str(res)
        + " words."
        + " Only the first 500 words will be reviewed."
    )

doc2 = doc2[:MAX_WORDS]
submit_button = st.button(label="Submit Ticket")

if submit_button:
    test = DataFrame({"short_description": [doc1], "long_description": [doc2]})
    prediction = model.predict(test)
    prediction = "IT" if prediction[0] == 1 else 'HR'
    st.write("Ticket Category Type: {}".format(prediction))