import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
import glob
from uuid import uuid4

# Function to load documents
def load_documents(files):
    combined_data = []
    for file in files:
        try:
            loader = Docx2txtLoader(file)
            data = loader.load()
            combined_data.append(data[0].page_content)
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    return " ".join(combined_data)

# Load document files
files = glob.glob("./GEN_AI/*.docx")
data_content = load_documents(files)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([data_content])

# Model for embedding
model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Create in-memory document store
document_store = InMemoryDocumentStore()

# Embedding pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
indexing_pipeline.connect("embedder.documents", "writer.documents")

textss = [Document(content=doc.page_content, id=str(uuid4()), meta=doc.metadata) for doc in texts]
indexing_pipeline.run({"documents": textss})

retriever = InMemoryEmbeddingRetriever(document_store=document_store)
reader = ExtractiveReader(model="deepset/roberta-base-squad2")
reader.warm_up()

# Create QA pipeline
extractive_qa_pipeline = Pipeline()
extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
extractive_qa_pipeline.add_component(instance=reader, name="reader")
extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

st.set_page_config(page_title="Document QA Chatbot", page_icon="🤖")
st.title("Document QA Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}  # Initial message
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the QA pipeline
    answer = extractive_qa_pipeline.run(
        data={"embedder": {"text": prompt}, "retriever": {"top_k": 5}, "reader": {"query": prompt, "top_k": 1}}
    )

    extracted_content = []
    for ans in answer['reader']['answers']:
        if ans.score > 0.5 and ans.data and ans.document:
            extracted_content.append(ans.document.content)

    if extracted_content:
        bot_answer = extracted_content[0]
    else:
        bot_answer = "I couldn't find relevant information, could you please rephrase your question?"

    st.session_state.messages.append({"role": "assistant", "content": bot_answer})

    with st.chat_message("assistant"):
        st.markdown(bot_answer)
