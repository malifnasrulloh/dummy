import streamlit as st
import utils
import numpy as np
import tensorflow as tf

utils.download_model()
model = utils.text_load_model()
tokenizer = utils.text_tokenizer()
label = ['negative', "neutral", "positive"]

st.title("Sentiment Analyst (Eng)")

text_inputs = st.text_area("Masukkan Teks", placeholder="I think Alip so cool...")

if text_inputs:
    text, results = utils.text_inference(model, tokenizer, text_inputs)
    for text, pred in zip(text, results):
        st.write({
            "text" : text,
            "result": label[pred]
        })