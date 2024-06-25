import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import os, sys

from utils import *
from ai_agents import *

def setup_sidebar():
    st.set_page_config(page_title="AI Agent with tools", page_icon="🚀")
    
    api_key = st.sidebar.text_input("API Key", type="password")
    
    model_choice = st.sidebar.radio(
        "Choose a model:", ("Llama3 - 8B", "gpt-4-0613 (Not Implemented)"))

    available_tools = {
        "Search": DuckDuckGoSearchRun(name="Search"),
    }

    # st.sidebar.text("Select tools:")
    # st.sidebar.checkbox("Search (DuckDuckGo) 🪿", value=True, disabled=True)

    # selected_tools = [available_tools["Search"]]

    st.sidebar.text("Select Options:")

    # Tool selections
    # if st.sidebar.checkbox("Upload a File 📓"):
    #     uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))
    #     available_tools['Uploaded Data'] = uploaded_file.read().decode()

    return api_key, model_choice, available_tools

def ai_message(msg: str):
    st.chat_message("assistant", avatar='./ui_imgs/assistant.jpeg').write(msg)

def main():
    prompt = None
    data = ""

    # Init the Utils class
    utils = Utils()

    # Load the LLM Model
    model = utils.load_model()
    ai_agents = AI_Agents(model, "./text_files")
    # Load the Chroma Database or Return Null if nothing is there in DB
    vectordb = utils.load_db()


    if not prompt:
        prompt = st.chat_input(placeholder="What would you like to know?")

    # uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))

    while True:
        prompt = input("Enter value:")
        # new_file_uploaded = True if uploaded_file else False
        
        # Classify the prompt into one of 5 categories
        prompt_result = ai_agents.prompt_classifier(prompt)

        # if prompt_result == "save something in memory":
        #     if new_file_uploaded:
        #         ai_message("Data uploaded")
        #         structured_output = ai_agents.extract_from_uploaded_file(uploaded_file)
        #         ai_message(structured_output['final_output'])
                
        #         if os.path.exists("extracted_info.txt"):
        #             with open("extracted_info.txt", "a+") as text_file:
        #                 text_file.write(structured_output['final_output'])
        #         else:
        #             with open("extracted_info.txt", "wb") as text_file:
        #                 text_file.write(structured_output['final_output'])
                
        #         # Store the Extracted Data in Database
        #         utils.store_in_db("extracted_info.txt")
        #         ai_message("Data stored in DB")
            
        #     else:
        #         ai_message("You have not uploaded any file to save in memory! \
        #                     Please Upload file and Enter New prompt")

        if prompt_result == "deduce memory from unstructured text":
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            if type(retriever) == str:
                print("There is Nothing in Memory")
            else:
                # create the chain to answer questions 
                qa_chain = RetrievalQA.from_chain_type(llm=model, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                return_source_documents=True)
                llm_response = qa_chain(prompt)['result']
                print(llm_response)

        elif prompt_result == "update memory":
            print("Loading Old Memory....Updating Memory")

            docs = vectordb.similarity_search(prompt)
            doc_id = docs[0].metadata['id']

            stored_memory = ai_agents.memory_management(prompt, docs[0])
            
            document = docs[0]
            document.page_content = stored_memory
            vectordb.update_document(doc_id, document)
            vectordb.persist()
            
            print("Updated DB with new data")

        elif prompt_result == "delete memory":
            print("Deleting memory")
            extracted_class = ai_agents.category_extraction(prompt)

            db_result = vectordb.similarity_search(extracted_class.lower())[0]
            db_index = db_result.metadata['id']
            updated_memory = ai_agents.delete_memory(prompt, db_result)
            print("Updated Memory ------------ \n\n")

            document = db_result
            
            if len(updated_memory)>0:
                document.page_content = updated_memory
                vectordb.update_document(db_index, document)
                vectordb.persist()

        elif prompt_result == "off_topic":
            response = utils.generic_response(model, prompt)
            print(response)

# Execute the main function
if __name__ == "__main__":
    main()