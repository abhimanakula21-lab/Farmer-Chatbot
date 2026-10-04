import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


# =========================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Farmer AI Assistant",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Farmer AI Assistant")

st.write(
    "Ask questions about crops, farming practices, pests, "
    "fertilizers and agricultural problems."
)


# =========================================================
# 3. LOAD AGRICULTURAL KNOWLEDGE + CREATE FAISS DATABASE
# =========================================================

@st.cache_resource
def create_vector_database():

    file_path = "data/agriculture.txt"

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    documents = loader.load()

    # Split agricultural information into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # Create sentence embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    vector_database = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_database


# =========================================================
# 4. INITIALIZE VECTOR DATABASE
# =========================================================

with st.spinner("Loading agricultural knowledge..."):

    vector_database = create_vector_database()


# =========================================================
# 5. LOAD GEMINI
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY
)


# =========================================================
# 6. RAG PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful agricultural assistant for farmers.

Answer the farmer's question using the provided agricultural
knowledge.

IMPORTANT RULES:

1. Use the retrieved agricultural information as the primary source.
2. Do not invent agricultural facts.
3. If the retrieved information is insufficient, clearly say:
   "The available agricultural knowledge does not contain enough
   information to answer this question."
4. Give simple and practical explanations.
5. Do not recommend dangerous or excessive pesticide or fertilizer use.
6. When treatment depends on crop, location, soil, weather,
   or disease identification, recommend consulting a local
   agricultural expert.
7. Do not pretend to diagnose a crop disease with certainty.
8. Keep the answer understandable for a farmer.

RETRIEVED AGRICULTURAL INFORMATION:

{context}

FARMER'S QUESTION:

{question}

ANSWER:
"""
)


# =========================================================
# 7. CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask your farming question..."
)


# =========================================================
# 8. PROCESS FARMER QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Display farmer's question
    # -----------------------------------------------------

    with st.chat_message("user"):
        st.write(question)


    # -----------------------------------------------------
    # Retrieve relevant agricultural information
    # -----------------------------------------------------

    with st.spinner("Searching agricultural knowledge..."):

        retriever = vector_database.as_retriever(
            search_kwargs={"k": 3}
        )

        retrieved_documents = retriever.invoke(question)


    # -----------------------------------------------------
    # Combine retrieved documents into context
    # -----------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )


    # -----------------------------------------------------
    # Create RAG prompt
    # -----------------------------------------------------

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )


    # -----------------------------------------------------
    # Generate answer using Gemini
    # -----------------------------------------------------

    with st.spinner("Thinking..."):

        try:

            response = llm.invoke(
                formatted_prompt
            )

        except Exception as e:

            st.error(
                f"Error while contacting Gemini: {e}"
            )

            st.stop()


    # -----------------------------------------------------
    # Extract only the actual text from Gemini
    # -----------------------------------------------------

    if isinstance(response.content, list):

        answer_parts = []

        for block in response.content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    answer_parts.append(
                        block.get("text", "")
                    )

            elif isinstance(block, str):

                answer_parts.append(block)

        answer = "\n".join(answer_parts).strip()

    else:

        answer = str(response.content).strip()


    # -----------------------------------------------------
    # Display Gemini answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)


        # -------------------------------------------------
        # Show retrieved sources
        # -------------------------------------------------

        with st.expander(
            "📚 Retrieved Agricultural Information"
        ):

            for i, document in enumerate(
                retrieved_documents,
                start=1
            ):

                st.write(
                    f"**Source {i}:**"
                )

                st.write(
                    document.page_content
                )