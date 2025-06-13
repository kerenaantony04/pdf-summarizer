import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

# 1. Setup API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDyuQQiJyDfCpayS7-sLPHZKtnAAZQRWEU"  # 🔑 Replace with your Gemini API key

# 2. UI Layout
st.set_page_config(page_title="📄 PDF Summarizer with Gemini", layout="wide")
st.title("📄 PDF Summarizer with LangChain + Gemini")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # 3. Load PDF
    st.info("Extracting text from PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # 4. Create vectorstore using embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(pages, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 5. Setup LLM and Memory
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Fixes ValueError from multiple output keys
    )

    # 6. Conversational RAG Chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    # 7. User Query Input
    st.success("✅ PDF successfully processed. You can now ask questions or type 'summarize this document'.")
    query = st.text_input("💬 Ask something or summarize the document")

    if query:
        with st.spinner("🧠 Generating answer..."):
            result = rag_chain.invoke({"question": query})
            st.markdown(f"**💬 Response:**\n\n> {result['answer']}")

    # 8. Show Chat History
    with st.expander("🧠 Conversation History"):
        for msg in memory.chat_memory.messages:
            role = "🧑 You" if msg.type == "human" else "🤖 Bot"
            st.markdown(f"**{role}:** {msg.content}")
