import streamlit as st
from datasets import load_dataset
from haystack import Document
from haystack.components.readers import ExtractiveReader

# Load the dataset
dataset = load_dataset("PedroCJardim/QASports", "basketball", split="validation")

# Load the model
reader = ExtractiveReader(model="laurafcamargos/distilbert-qasports-basket-small")
reader.warm_up()

# Running using the Reader
docs = [
    Document(content="Paris is the capital of France."),
    Document(content="Berlin is the capital of Germany.")
]

query = "What is the capital of France?"
answer = reader.run(query="What is the capital of France?", documents=docs, top_k=1)

st.json(answer)
