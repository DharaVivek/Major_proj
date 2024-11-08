import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from each page of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_web_text(url):
    """Scrape the main text content from a website."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('div', {'class': 'text'})
        if main_content:
            paragraphs = main_content.find_all('p', {'dir': 'ltr'})
            text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
            return text
        else:
            return "Main content not found on the webpage."
    except Exception as e:
        return f"Error fetching web content: {e}"

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate embeddings for the text chunks and store them in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Define the conversational chain with custom prompts."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context".

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Fetch similar documents based on the user query and generate an answer."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config("Chat PDF & Web")
    st.header("Query PDFs and Web Content")

    user_question = st.text_input("Ask a Question about the Uploaded Files or Web Content")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        # PDF Upload Section
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        # URL Input Section for Multiple Web Content
        urls = st.text_area("Enter URLs to scrape content from (one per line)")

        # Process Button
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process PDF files if uploaded
                if pdf_docs:
                    raw_pdf_text = get_pdf_text(pdf_docs)
                    pdf_text_chunks = get_text_chunks(raw_pdf_text)
                    get_vector_store(pdf_text_chunks)
                    st.success("PDF Processing Complete")

                # Process Web content if URLs are provided
                if urls.strip():
                    all_web_text = ""
                    for url in urls.splitlines():
                        url = url.strip()
                        if url:
                            web_text = get_web_text(url)
                            if "Main content not found" not in web_text and "Error" not in web_text:
                                all_web_text += web_text + "\n\n"
                            else:
                                st.warning(f"Failed to retrieve content from: {url}")

                    if all_web_text:
                        web_text_chunks = get_text_chunks(all_web_text)
                        get_vector_store(web_text_chunks)
                        st.success("Web Content Processing Complete")

if __name__ == "__main__":
    main()
