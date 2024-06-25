import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import os, time

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

    st.sidebar.text("Select Options:")

    # Tool selections
    # if st.sidebar.checkbox("Upload a File 📓"):
    #     uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))
    #     available_tools['Uploaded Data'] = uploaded_file.read().decode()

    return api_key, model_choice, available_tools

def ai_message(msg: str):
    st.chat_message("assistant", avatar='./ui_imgs/assistant.jpeg').write(msg)

def main():
    api_key, model_choice, tools = setup_sidebar()
    prompt = None
    data = ""

    # Init the Utils class
    utils = Utils()

    progress_text = "Please Enter API Key of Groq"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        if not api_key:
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            continue
        else:
            my_bar.empty()
            break
        
    # Load the LLM Model
    model = utils.load_model(api_key)
    ai_agents = AI_Agents(model, "./text_files")
    # Load the Chroma Database or Return Null if nothing is there in DB
    vectordb = utils.load_db()

    st.title("🚀 AI Assistant")
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="memory", output_key="output"
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}
        prompt = None 

    if not prompt:
        prompt = st.chat_input(placeholder="What would you like to know?")

    uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))

    if prompt:
        st.chat_message("user", avatar='./ui_imgs/user.jpeg').write(prompt)
        if not api_key:
            st.info("Please add your Model API key to continue.")
            st.stop()
        new_file_uploaded = True if uploaded_file else False
        
        # Classify the prompt into one of 5 categories
        prompt_result = ai_agents.prompt_classifier(prompt)

        if prompt_result == "save something in memory":
            if new_file_uploaded:
                ai_message("Data uploaded")
                structured_output = ai_agents.extract_from_uploaded_file(uploaded_file)
            
                utils.store_in_db(structured_output)
                ai_message("Data stored in DB")
            else:
                ai_message("You have not uploaded any file to save in memory! \
                            Please Upload file and Enter New prompt")

        elif prompt_result == "deduce memory from unstructured text":
            ai_message(prompt)
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            if type(retriever) == str:
                ai_message("There is Nothing in Memory")
            else:
                # create the chain to answer questions 
                qa_chain = RetrievalQA.from_chain_type(llm=model, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                return_source_documents=True)
                llm_response = qa_chain(prompt)['result']
                ai_message(llm_response)

        elif prompt_result == "update memory":
            ai_message("Loading Old Memory....Updating Memory")

            docs = vectordb.similarity_search(prompt)
            doc_id = docs[0].metadata['id']

            stored_memory = ai_agents.memory_management(prompt, docs[0])
            
            document = docs[0]
            document.page_content = stored_memory
            vectordb.update_document(doc_id, document)
            vectordb.persist()
            
            ai_message("Updated DB with new data")

        elif prompt_result == "delete memory":
            
            extracted_class = ai_agents.category_extraction(prompt)

            db_result = vectordb.similarity_search(extracted_class.lower())[0]
            db_index = db_result.metadata['id']

            ai_message("Deleting Memory")
            updated_memory = ai_agents.delete_memory(prompt, db_result)
            document = db_result
            if len(updated_memory)>0:
                document.page_content = updated_memory
                vectordb.update_document(db_index, document)
                vectordb.persist()

        elif prompt_result == "off_topic":
            with st.chat_message("assistant", avatar='./ui_imgs/assistant.jpeg'):
                response = utils.generic_response(model, prompt)
                try:
                    st.write(response)
                except Exception as e:
                    st.error("Something went wrong. Please try again later.")
                    msgs.clear()
                    msgs.add_ai_message("How can I help you?")


# Execute the main function
if __name__ == "__main__":
    main()