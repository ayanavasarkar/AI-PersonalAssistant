from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import os, textwrap

from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from chroma_aya import Chroma_AYA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util


class Utils():
    """
    A utility class providing various functions for handling language models, 
    text processing, and database operations.
    """

    def __init__(self) -> None:
        """
        Initializes the Utils class with an embedding model from HuggingFace.
        """
        self.embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    def load_model(self, api_key: str) -> ChatGroq:
        """
        Loads the language model using the provided API key.

        Args:
            api_key (str): The API key for authentication.

        Returns:
            ChatGroq: An instance of the ChatGroq model.
        """
        # os.environ.get("GROQ_API_KEY")
        model = ChatGroq(
                api_key= api_key,
                model="llama3-70b-8192"
            )
        return model

    def generic_response(self, model: ChatGroq, prompt: str) -> str:
        """
        Generates a response to the given prompt using the specified model (LLM)

        Args:
            model: The language model to be used for generating the response.
            prompt (str): The input prompt for the model.

        Returns:
            str: The generated response content.
        """
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Prepare the messages for the model
        messages = [("system", "You are a helpful assistant. Respond to the user sentence in English."),
            ("human", prompt),
        ]

        # Invoke the model with the prepared messages
        response = model.invoke(messages)
        return response.content

    def load_db(self, path = "/tmp/db") -> Chroma_AYA:
        """
        Loads the vector database from the specified path, or returns a message if the path does not exist.

        Args:
            path (str): The path to the vector database directory.

        Returns:
            Chroma_AYA: The loaded vector database or a message indicating absence of memory.
        """
        if os.path.exists(path):
            # Loading the preexisting database
            vectordb = Chroma_AYA(
                persist_directory=path, 
                embedding_function=self.embedding
            )
        else:
            return "Nothing in Memory"
        
        # Persist the database to disk
        vectordb.persist()
        return vectordb

    def text_splitter(self, text: list) -> list:
        """
        Splits the given text into smaller chunks using a recursive character text splitter.

        Args:
            text (list): The text to be split.

        Returns:
            list: A list of split text chunks.
        """
        #splitting the text into
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(text)
        return texts

    def compare_new_data_to_db(self, splitted_texts):
        """
        # TODO: Add This Function for comparing memories and only adding the necessary part.
        Compares new data to the existing data in the database using sentence embeddings.

        Args:
            splitted_texts (list): List of text chunks to be compared.
        """
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        #Compute embedding for both lists
        # embedding_1= model.encode(texts[0].page_content, convert_to_tensor=True)
        # embedding_2 = model.encode(t_da, convert_to_tensor=True)

        # util.pytorch_cos_sim(embedding_1, embedding_2)
        # for each_split in splitted_texts:
        #     page_content = each_split.page_content
        # pass

    def store_in_db(self, data: str) -> None:
        """
        Stores the provided data in the vector database, splitting it into chunks if necessary.

        Args:
            data (str): The data to be stored.
        """
        persist_directory = '/tmp/db'
        # loader = TextLoader(data_path)
        # documents = loader.load()
        splitted_texts = self.text_splitter([Document(page_content=data, metadata={})])

        # new_texts = self.compare_new_data_to_db(splitted_texts)

        if os.path.exists(persist_directory):
            # Loading the preexisting DB
            vectordb = Chroma_AYA(persist_directory=persist_directory, 
                    embedding_function=self.embedding)
            vectordb.add_documents(documents=splitted_texts)
        
        else:
            vectordb = Chroma_AYA.from_documents(documents=splitted_texts, 
                                    embedding=self.embedding,
                                    persist_directory=persist_directory)
        
        # persiste the db to disk
        vectordb.persist()

    def wrap_text_preserve_newlines(self, text: str, width=110) -> str:
        """
        Wraps the text while preserving newlines.

        Args:
            text (str): The text to be wrapped.
            width (int): The width at which to wrap the text.

        Returns:
            str: The wrapped text with preserved newlines.
        """
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(self, llm_response: dict) -> str:
        """
        Processes the LLM response by wrapping the text while preserving newlines.

        Args:
            llm_response (dict): The response from the LLM.

        Returns:
            str: The processed response text.
        """
        return self.wrap_text_preserve_newlines(llm_response['result'])