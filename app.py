import streamlit as st
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from utils import get_unique_docs

# Load the dataset
unique_docs = set()
dataset = load_dataset("PedroCJardim/QASports", "basketball")
docs_validation = get_unique_docs(dataset["validation"], unique_docs)
docs_train = get_unique_docs(dataset["train"], unique_docs)
docs_test = get_unique_docs(dataset["test"], unique_docs)
docs_all = docs_validation + docs_train + docs_test

# Create the Question Answering pipeline
# Create in memory database
document_store = InMemoryDocumentStore()
document_store.write_documents(documents=docs_all)
# Create the retriever and reader
retriever = InMemoryBM25Retriever(document_store=document_store)
reader = ExtractiveReader(model="laurafcamargos/distilbert-qasports-basket-small")
reader.warm_up()
# Create the pipeline
pipe = Pipeline()
pipe.add_component(instance=retriever, name="retriever")
pipe.add_component(instance=reader, name="reader")
pipe.connect("retriever.documents", "reader.documents")

# Streamlit interface
st.markdown("""This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football. It counts over 1.5 million questions and answers about 54k preprocessed, cleaned and organized documents from Wikipedia-like sources.""")
st.subheader('QASports: Basketball', divider='rainbow')

top_k = 3
user_query = None
user_query = st.text_input("Please, make a question about basketball:")

if user_query:
    answer = pipe.run(data={
        "retriever": {"query": user_query, "top_k": 10},
        "reader": {"query": user_query, "top_k": top_k},
    })
    # Display only the top k answers
    st.json(answer["reader"]["answers"][0:top_k])
