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

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
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

with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])

    with c2:
        doc1 = st.text_area(
            "Paste your text below (max 50 words)",
            height=150,
        )

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

        doc2 = st.text_area(
            "Paste your text below (max 500 words)",
            height=350,
        )

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

        submit_button = st.form_submit_button(label="Submit Ticket")

if not submit_button:
    st.stop()

test = DataFrame({"short_description": [doc1], "long_description": [doc2]})
prediction = model.predict(test)
prediction = "IT" if prediction[0] == 1 else 'HR'

st.write(prediction)