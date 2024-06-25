from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import os, textwrap

from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from chroma_aya import Chroma_AYA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Utils():

    def __init__(self) -> None:
        self.embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    def load_model(self):
        model = ChatGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
                model="llama3-70b-8192"
            )
        return model

    def generic_response(self, model, prompt):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        messages = [("system", "You are a helpful assistant. Respond to the user sentence in English."),
            ("human", prompt),
        ]
        response = model.invoke(messages)
        return response.content

    def load_db(self, path = "/tmp/db"):
        
        if os.path.exists(path):
            # Loading the preexisting DB
            print("Loading DB....")
            vectordb = Chroma_AYA(persist_directory=path, 
                    embedding_function=self.embedding)
            
        else:
            return "Nothing in Memory"
        
        # persiste the db to disk
        vectordb.persist()
        return vectordb

    def text_splitter(self, text):
        #splitting the text into
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(text)
        return texts

    def store_in_db(self, data_path):
        
        persist_directory = '/tmp/db'

        loader = TextLoader(data_path)
        documents = loader.load()
        splitted_texts = self.text_splitter(documents)

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

    def wrap_text_preserve_newlines(self, text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(self, llm_response):
        return self.wrap_text_preserve_newlines(llm_response['result'])