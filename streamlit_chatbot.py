############################################################################################

#                                  Author: Anass MAJJI                                     #
 
#                               File Name: streamlit_app.py                                #

#                           Creation Date: May 06, 2024                                    #

#                         Source Language: Python                                          #

#         Repository:    https://github.com/amajji/LLM-RAG-Chatbot-With-LangChain          #

#                              --- Code Description ---                                    #

#    Deploy LLM RAG Chatbot with Langchain on a Streamlit web application using only CPU   #

############################################################################################


############################################################################################
#                                   Packages                                               #
############################################################################################


# Import Python Libraries
import streamlit as st
import logging
import logging.config
import os
#import torch
import time
import psutil
from memory_profiler import profile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory


#########################################################################################
#                                Variables                                              #
#########################################################################################

st.set_page_config(layout="wide")
#STREAMLIT_STATIC_PATH = str(pathlib.Path(st.__path__[0]) / "AI_Hackathon_Dataset/pdf")
#STREAMLIT_STATIC_PATH = "/app/dataset/pdf"
#STREAMLIT_STATIC_PATH = "./dataset/pdf"

static_path = os.getenv("STREAMLIT_STATIC_PATH")
    
# Configure logging from the log.ini file
# "/app/log.ini"
# Get the absolute path to the log.ini file inside the container
log_config_path = "/app/app/log.ini"

# Check if the file exists
if os.path.exists(log_config_path):
    logging.config.fileConfig(log_config_path)
else:
    print(f"Error: {log_config_path} not found!")

# Get the logger
logger = logging.getLogger()
#########################################################################################
#                                Functions                                              #
#########################################################################################

#st.cache_resource.clear()

# def get_memory_usage():
#     """
#     Function to track and print CPU and GPU memory usage
#     """

#     # Get CPU memory usage
#     memory = psutil.virtual_memory()
#     cpu_memory = memory.percent  # Memory usage percentage of the system

#     # Get GPU memory usage
#     if torch.cuda.is_available():
#         gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
#         gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # in MB
#         gpu_memory = {
#             "allocated_memory": gpu_memory_allocated,
#             "reserved_memory": gpu_memory_reserved,
#         }
#     else:
#         gpu_memory = None
#     return cpu_memory, gpu_memory
 



st.cache_resource(max_entries=1)
@profile
def create_vector_db(data_path):
    """function to create vector db provided the pdf files"""

    logger.info(" -- loader ... " )
    # define the docs's path
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    logger.info(" -- loader OK ")

    logger.info(" -- documents ... " )
    # load documents
    documents = loader.load()
    logger.info(" -- documents OK " )

    # use recursive splitter to split each document into chunks
    logger.info(" -- text_splitter ... " )
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    logger.info(" -- text_splitter OK " )

    logger.info(" -- texts ... " )
    texts = text_splitter.split_documents(documents)
    logger.info(" -- texts OK " )   

    # Initialize embeddings model with GPU support
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # generate embeddings for each chunk
    logger.info(" -- embeddings ... " )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": device},
    )
    logger.info(" -- embeddings OK " )
    
    logger.info(" -- FAISS ... " )
    # indexing database
    db = FAISS.from_documents(texts, embeddings)
    logger.info(" -- FAISS OK " )
    return db




st.cache_resource(max_entries=1)
@profile
def load_llm(temperature, max_new_tokens, top_p, top_k):
    """Load the LLM model"""
    
    logger.info(" -- Load llm model with CTransformers ...")
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    logger.info(" -- Load llm model with CTransformers OK ")

    # List all available attributes for the model
    logger.info(llm.config)  

    # return the LLM
    return llm



st.cache_resource
@profile
def model_retriever(vector_db):
    # Create a retriever object from the 'db' with a search configuration where
    # it retrieves up to top_k relevant splits/documents.
    logger.info(" -- Build retriever ...")
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})   
    logger.info(" -- Build retriever OK ") 

    return retriever


st.cache_data
@profile
def q_a_llm_model(retriever, llm_model):
    """
    This function loads the LLM model, gets the relevent
    docs for a given query and provides an answer
    """

    # Create a question-answering instance (qa) using the RetrievalQA class.
    # It's configured with a language model (llm), a chain type "refine,"
    # the retriever we created, and an option to not return source documents.
    q_a = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=False,
    )

    return q_a




#########################################################################################
#                                Main code                                              #
#########################################################################################


# First streamlit's page
def page_1():

    # define the title
    st.title("✨ LLM with RAG")

    # quick decription of the webapp
    st.markdown(
        """
        This interactive dashboard allows users to extract information from external documents seamlessly. 
        Powered by the Llama 2-7B model and LangChain for retrieval-augmented generation (RAG), the app enables users to ask questions, 
        and the LLM delivers relevant answers based on the available documents.

        To enhance performance, we've optimized the model using the GGML quantization technique, reducing inference 
        time while maintaining accuracy. Notably, the application is designed to work efficiently even on CPU processors.
        """
    )

    st.markdown(
        """
        Below, a chat to interact with the LLM.
        """
    )

    # Text generation params
    st.sidebar.subheader("Text generation parameters")
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "Top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )
    top_k = st.sidebar.slider("Top_k", min_value=0, max_value=100, value=20, step=10)
    max_length = st.sidebar.slider(
        "Max_length", min_value=64, max_value=4096, value=512, step=8
    )





    # Lazy load the vector DB and LLM only when needed
    if "vector_db" not in st.session_state:
        # Check memory usage before the forward pass
        logger.info("Creation of the vector database ... ")
        start_time = time.time()

        # Lazy load the vector database (FAISS index) when needed
        st.session_state["vector_db"] = create_vector_db(static_path)

        # end time
        end_time = time.time()
        logger.info(f" Creation of the vector database in {end_time - start_time: .2f} seconds ")

    #cpu_memory, gpu_memory = get_memory_usage()
    #logger.info(f"CPU Memory Usage: {cpu_memory}%")

    # if gpu_memory:
    #     logger.info(f"GPU Memory Allocated: {gpu_memory['allocated_memory']} MB")
    #     logger.info(f"GPU Memory Reserved: {gpu_memory['reserved_memory']} MB")


    if "llm_model" not in st.session_state:
        logger.info(" Model loading ... ")
        start_time = time.time()

        # Lazy load the language model when needed
        st.session_state["llm_model"] = load_llm(temperature, max_length, top_p, top_k)

        # end time
        end_time = time.time()
        logger.info(f" Model loading in {end_time - start_time: .2f} seconds ")

    # cpu_memory, gpu_memory = get_memory_usage()
    # logger.info(f"CPU Memory Usage: {cpu_memory}%")

    # if gpu_memory:
    #     logger.info(f"GPU Memory Allocated: {gpu_memory['allocated_memory']} MB")
    #     logger.info(f"GPU Memory Reserved: {gpu_memory['reserved_memory']} MB")



    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.caption("🚀 A Streamlit chatbot powered by OpenAI")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get the vector database and LLM model from session state
        vector_db = st.session_state["vector_db"]
        llm_model = st.session_state["llm_model"]

        # chat history 
        #chat_history = []

        # Model retriever should ideally be done only once and stored in session_state if not done already
        if "retriever" not in st.session_state:
            st.session_state["retriever"] = model_retriever(vector_db)

        # Use the retriever that is already created and stored
        retriever = st.session_state["retriever"]

        # Create the Q&A model (you could cache this too, or ensure it's created only once)
        if "q_a" not in st.session_state:
            st.session_state["q_a"] = q_a_llm_model(retriever, llm_model)

        # Use the stored Q&A model for the query
        q_a = st.session_state["q_a"]
        
        # Start the timer
        start_time = time.time()

        # Get the result from the Q&A model
        result = q_a.run({"query": prompt})

        # End the timer
        end_time = time.time()

        # Calculate the inference time
        inference_time = end_time - start_time

        logger.info(f"Inference Time: {inference_time:.4f} seconds")

        st.session_state.messages.append(
            {"role": "assistant", "content": "voici le message de retour"}
        )
        st.chat_message("assistant").write(result)



def main():

    """A streamlit app template"""
    st.sidebar.title("Menu")

    PAGES = {
        "🎈 LLaMA2-7B LLM": page_1,
    }

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Select your page : ", list(PAGES.keys()))
    #st.sidebar_caption()

    PAGES[selection]()

    st.sidebar.title("About")
    st.sidebar.info(
        """
    Web App URL: <https://amajji-streamlit-dash-streamlit-app-8i3jn9.streamlit.app/>
    GitHub repository: <https://github.com/amajji/LLM-RAG-Chatbot-With-LangChain>
    """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    MAJJI Anass 
    [GitHub](https://github.com/amajji) | [LinkedIn](https://fr.linkedin.com/in/anass-majji-729773157)
    """
    )


if __name__ == "__main__":
    main()
