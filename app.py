import streamlit as st
from transformers import pipeline

pipe = pipeline("sentiment-analysis")
text = st.text_area("Enter some text")

if text:
    result = pipe(text)
    st.json(result)
