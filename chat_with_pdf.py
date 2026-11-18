import tempfile
from langchain_couchbase import CouchbaseSearchVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA


# Every setting the app needs once the user is logged in. Add new settings here
# so they are validated before any client is constructed.
REQUIRED_ENV_VARS = (
    "NVIDIA_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
    "INDEX_NAME",
)


def find_missing_settings(names, environ=None):
    """Return the names that are unset or empty, in the order given"""
    environ = os.environ if environ is None else environ
    return [name for name in names if not environ.get(name)]


def load_settings(names):
    """Return {name: value} for the given environment variables.

    Shows one error listing every missing variable and stops the app if any
    are unset or empty.
    """
    missing = find_missing_settings(names)
    if missing:
        st.error(
            f"Missing environment variables: {', '.join(missing)}. "
            "Please add them to the secrets.toml file"
        )
        st.stop()
    return {name: os.environ[name] for name in names}


def save_to_vector_store(uploaded_file, vector_store):
    """Chunk the PDF & store it in Couchbase Vector Store"""
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load only after the file is closed so the PDF is fully written
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )

        doc_pages = text_splitter.split_documents(docs)

        vector_store.add_documents(doc_pages)
        st.info(f"PDF loaded into vector store in {len(doc_pages)} documents")


@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    _embedding,
    index_name,
):
    """Return the Couchbase vector store"""
    vector_store = CouchbaseSearchVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=_embedding,
        index_name=index_name,
    )
    return vector_store


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    from couchbase.options import ClusterOptions
    from datetime import timedelta

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    options.apply_profile("wan_development")
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


if __name__ == "__main__":
    # Authorization
    if "auth" not in st.session_state:
        st.session_state.auth = False

    st.set_page_config(
        page_title="Chat with your PDF using Langchain, Couchbase, Nvidia NIM and Meta LLama3",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    AUTH = load_settings(["LOGIN_PASSWORD"])["LOGIN_PASSWORD"]

    # Authentication
    user_pwd = st.text_input("Enter password", type="password")
    pwd_submit = st.button("Submit")

    if pwd_submit and user_pwd == AUTH:
        st.session_state.auth = True
    elif pwd_submit and user_pwd != AUTH:
        st.error("Incorrect password")

    if st.session_state.auth:
        # Load and validate all settings in one place
        settings = load_settings(REQUIRED_ENV_VARS)

        # Use Nvidia's embeddings model
        embedding = NVIDIAEmbeddings(
            nvidia_api_key=settings["NVIDIA_API_KEY"],
            model="nvidia/nv-embed-v1")

        # Connect to Couchbase Vector Store
        cluster = connect_to_couchbase(
            settings["DB_CONN_STR"], settings["DB_USERNAME"], settings["DB_PASSWORD"]
        )

        vector_store = get_vector_store(
            cluster,
            settings["DB_BUCKET"],
            settings["DB_SCOPE"],
            settings["DB_COLLECTION"],
            embedding,
            settings["INDEX_NAME"],
        )

        # Use couchbase vector store as a retriever for RAG
        retriever = vector_store.as_retriever()

        # Build the prompt for the RAG
        template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        # Use Meta's LLama3 as the LLM for the RAG using NVIDIA NIM's API
        llm = ChatNVIDIA(
            nvidia_api_key=settings["NVIDIA_API_KEY"],
            temperature=0,
            model="meta/llama3-70b-instruct",
        )

        # RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )


        # Frontend
        couchbase_logo = (
            "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"
        )

        st.title("Chat with PDF")

        with st.sidebar:
            st.header("Upload your PDF")
            with st.form("upload pdf"):
                uploaded_file = st.file_uploader(
                    "Choose a PDF.",
                    help="The document will be deleted after one hour of inactivity (TTL).",
                    type="pdf",
                )
                submitted = st.form_submit_button("Upload")
                if submitted:
                    # store the PDF in the vector store after chunking
                    save_to_vector_store(uploaded_file, vector_store)

            st.subheader("How does it work?")
            st.markdown(
                """
                For each question, you will get one answer generated using RAG from Couchbase Vector Search
                """
            )

            st.markdown(
                "For RAG, we are using [Langchain](https://langchain.com/), [Couchbase Vector Search](https://couchbase.com/), [NVIDIA NIM](https://build.nvidia.com) and [Meta LLAMA 3](https://llama.meta.com/llama3/). We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store."
            )

            # View Code
            if st.checkbox("View Code"):
                st.write(
                    "View the code here: [Github](https://github.com/couchbase-examples/rag-demo/blob/main/chat_with_pdf.py)"
                )

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?",
                    "avatar": "🤖",
                }
            )

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        # React to user input
        if question := st.chat_input("Ask a question based on the PDF"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)

            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": question, "avatar": "👤"}
            )

            # Add placeholder for streaming the response
            with st.chat_message("assistant", avatar=couchbase_logo):
                message_placeholder = st.empty()

            # stream the response from the RAG
            rag_response = ""
            for chunk in chain.stream(question):
                rag_response += chunk
                message_placeholder.markdown(rag_response + "▌")

            message_placeholder.markdown(rag_response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rag_response,
                    "avatar": couchbase_logo,
                }
            )
