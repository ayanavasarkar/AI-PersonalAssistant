import pytest
from unittest.mock import patch, MagicMock
from utils import Utils
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chroma_aya import Chroma_AYA
from langchain_core.documents import Document

@pytest.fixture
def utils():
    return Utils()

@patch('utils.ChatGroq')
def test_load_model(mock_chatgroq, utils):
    api_key = 'test_api_key'
    mock_model = MagicMock(spec=ChatGroq)
    mock_chatgroq.return_value = mock_model

    model = utils.load_model(api_key)
    mock_chatgroq.assert_called_once_with(api_key=api_key, model="llama3-70b-8192")
    assert model == mock_model

@patch('utils.StreamlitCallbackHandler')
@patch('utils.ChatGroq')
def test_generic_response(mock_chatgroq, mock_st_cb, utils):
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = 'test response'
    mock_model.invoke.return_value = mock_response
    mock_chatgroq.return_value = mock_model

    prompt = "Hello, how are you?"
    response = utils.generic_response(mock_model, prompt)
    mock_model.invoke.assert_called_once()
    assert response == 'test response'

@patch('os.path.exists')
@patch('utils.Chroma_AYA')
def test_load_db(mock_chroma_aya, mock_path_exists, utils):
    mock_path_exists.return_value = True
    mock_db = MagicMock(spec=Chroma_AYA)
    mock_chroma_aya.return_value = mock_db

    db_path = "/tmp/db"
    vectordb = utils.load_db(db_path)
    mock_chroma_aya.assert_called_once_with(persist_directory=db_path, embedding_function=utils.embedding)
    assert vectordb == mock_db

    mock_path_exists.return_value = False
    vectordb = utils.load_db(db_path)
    assert vectordb == "Nothing in Memory"

def test_text_splitter(utils):
    text = [Document(page_content="This is a test document.", metadata={})]
    split_texts = utils.text_splitter(text)
    assert isinstance(split_texts, list)

@patch('os.path.exists')
@patch('utils.Chroma_AYA')
def test_store_in_db(mock_chroma_aya, mock_path_exists, utils):
    data = "This is some test data."
    mock_path_exists.return_value = True
    mock_db = MagicMock(spec=Chroma_AYA)
    mock_chroma_aya.return_value = mock_db

    utils.store_in_db(data)
    mock_chroma_aya.assert_called()

def test_wrap_text_preserve_newlines(utils):
    text = "This is a test text.\nWith a newline."
    wrapped_text = utils.wrap_text_preserve_newlines(text, width=10)
    assert "\n" in wrapped_text

def test_process_llm_response(utils):
    llm_response = {'result': "This is a test response from the LLM."}
    processed_response = utils.process_llm_response(llm_response)
    assert processed_response == utils.wrap_text_preserve_newlines(llm_response['result'])
