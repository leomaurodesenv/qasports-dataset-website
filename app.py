import streamlit as st
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from utils import get_unique_docs


# Load the dataset
@st.cache_data(show_spinner=False)
def load_documents():
    """
    Load the documents from the dataset considering only unique documents.

    Returns:
    - documents: list of dictionaries with the documents.
    """
    unique_docs = set()
    dataset_name = "PedroCJardim/QASports"
    dataset_split = "basketball"
    st.caption(f'Fetching "{dataset_name}" dataset')
    # build the dataset
    dataset = load_dataset(dataset_name, dataset_split)
    docs_validation = get_unique_docs(dataset["validation"], unique_docs)
    docs_train = get_unique_docs(dataset["train"], unique_docs)
    docs_test = get_unique_docs(dataset["test"], unique_docs)
    documents = docs_validation + docs_train + docs_test
    return documents


@st.cache_data(show_spinner=False)
def get_document_store(documents):
    """
    Index the files in the document store.

    Args:
    - files: list of dictionaries with the documents.
    """
    # Create in memory database
    st.caption(f"Building the Document Store")
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents=documents)
    return document_store


@st.cache_data(show_spinner=False)
def get_question_pipeline(_doc_store):
    """
    Create the pipeline with the retriever and reader components.

    Args:
    - doc_store: instance of the document store.

    Returns:
    - pipe: instance of the pipeline.
    """
    st.caption(f"Building the Question Answering pipeline")
    # Create the retriever and reader
    retriever = InMemoryBM25Retriever(document_store=_doc_store)
    reader = ExtractiveReader(model="laurafcamargos/distilbert-qasports-basket-small")
    reader.warm_up()
    # Create the pipeline
    pipe = Pipeline()
    pipe.add_component(instance=retriever, name="retriever")
    pipe.add_component(instance=reader, name="reader")
    pipe.connect("retriever.documents", "reader.documents")
    return pipe


# # Create the retriever and reader
# retriever = InMemoryBM25Retriever(document_store=document_store())
# reader = ExtractiveReader(model="laurafcamargos/distilbert-qasports-basket-small")
# reader.warm_up()
# # Create the pipeline
# pipe = Pipeline()
# pipe.add_component(instance=retriever, name="retriever")
# pipe.add_component(instance=reader, name="reader")
# pipe.connect("retriever.documents", "reader.documents")

# Streamlit interface
with st.status(
    "Downloading dataset...", expanded=st.session_state.get("expanded", True)
) as status:
    documents = load_documents()
    status.update(label="Indexing documents...")
    doc_store = get_document_store(documents)
    status.update(label="Creating pipeline...")
    pipe = get_question_pipeline(doc_store)
    status.update(
        label="Download and indexing complete!", state="complete", expanded=False
    )
    st.session_state["expanded"] = False

st.subheader("🔎 QASports: Basketball", divider="rainbow")
st.caption(
    """This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football. It counts over 1.5 million questions and answers about 54k preprocessed, cleaned and organized documents from Wikipedia-like sources."""
)

if user_query := st.text_input(
    label="What do you want to know about Basketball?",
    placeholder="How many field goals did Kobe Bryant score?",
):
    try:
        top_k = 3
        answer = pipe.run(
            data={
                "retriever": {"query": user_query, "top_k": 10},
                "reader": {"query": user_query, "top_k": top_k},
            }
        )
        # Display only the top k answers
        max_k = len(answer["reader"]["answers"])
        st.json(answer["reader"]["answers"][0 : min(top_k, max_k)])
    except Exception as e:
        st.error(f"Error: We do not have an answer for your question.")
