---

# Document QA Chatbot

This project implements a Document Question-Answering (QA) chatbot using Streamlit and Haystack. The application allows users to upload and query `.docx` documents. It processes the documents, splits them into manageable chunks, and uses embedding techniques to provide accurate answers to user queries.

## Features

- Upload multiple `.docx` documents.
- Ask questions related to the content of the uploaded documents.
- Provides relevant answers based on the document's content.
- Error handling for unsupported or corrupt document files.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **Haystack**: For implementing the document processing and QA pipeline.
- **Langchain Community**: For loading `.docx` files.
- **Sentence Transformers**: For embedding documents and queries.
- **UUID**: For generating unique document IDs.

## Installation

1. Clone the repository or download the code files.
2. Ensure you have Python installed on your machine.
3. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the URL provided (typically `http://localhost:8501`).
3. Upload your `.docx` files using the interface.
4. Ask questions about the content in the uploaded documents using the chat interface.

## File Structure

```
/your_project_directory/
│
├── app.py                     # Main application file
└── /GEN_AI/                   # Directory for .docx files
    └── your_document.docx     # Example document file
```

## How It Works

1. **Loading Documents**: The application uses `Docx2txtLoader` from Langchain to load `.docx` files and extract their content.
2. **Text Splitting**: The content is split into chunks using `RecursiveCharacterTextSplitter`, which allows for efficient processing and retrieval of information.
3. **Embedding Pipeline**: The documents are embedded using the `SentenceTransformersDocumentEmbedder`. The embedded documents are stored in an in-memory document store.
4. **Retrieval and QA Pipeline**: A retrieval pipeline is created to find relevant chunks based on the user's queries. The `ExtractiveReader` component is used to provide answers to the queries.
5. **User Interface**: Streamlit is used to create a user-friendly interface for uploading documents and interacting with the chatbot.
---
