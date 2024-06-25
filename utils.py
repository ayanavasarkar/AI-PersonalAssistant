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

    def __init__(self) -> None:
        self.embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    def load_model(self, api_key):
        # os.environ.get("GROQ_API_KEY")
        model = ChatGroq(
                api_key= api_key,
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

    def compare_new_data_to_db(self, splitted_texts):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        #Compute embedding for both lists
        # embedding_1= model.encode(texts[0].page_content, convert_to_tensor=True)
        # embedding_2 = model.encode(t_da, convert_to_tensor=True)

        # util.pytorch_cos_sim(embedding_1, embedding_2)
        # for each_split in splitted_texts:
        #     page_content = each_split.page_content
        # pass

    def store_in_db(self, data_path):
        
        persist_directory = '/tmp/db'
        # loader = TextLoader(data_path)
        # documents = loader.load()
        splitted_texts = self.text_splitter([Document(page_content=data_path, metadata={})])

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