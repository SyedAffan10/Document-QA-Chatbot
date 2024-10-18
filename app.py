import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import glob

def load_documents(files):
    combined_data = []
    for file in files:
        try:
            loader = Docx2txtLoader(file)
            data = loader.load()
            for doc in data:
                combined_data.append(doc.page_content)
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    return combined_data

files = glob.glob("./GEN_AI/*.docx")
data_contents = load_documents(files)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50
)

texts = []
for content in data_contents:
    texts.extend(text_splitter.create_documents([content]))

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

vector_store = FAISS.from_texts([doc.page_content for doc in texts], embedding_model)

def retrieve_documents(query):
    docs = vector_store.similarity_search(query, k=5)
    return docs

def answer_question(query):
    docs = retrieve_documents(query)
    if not docs:
        return "I couldn't find relevant information. Could you rephrase your question?"

    llm = pipeline("text2text-generation", model="google/flan-t5-large")

    retriever_qa = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=llm),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        input_key="query"
    )

    answer = retriever_qa.run(query)
    return answer

st.set_page_config(page_title="GEN AI product knowledge", page_icon="🤖")
st.title("GEN AI product knowledge")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    bot_answer = answer_question(prompt)

    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
    with st.chat_message("assistant"):
        st.markdown(bot_answer)
