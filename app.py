
#External librearies 
import streamlit as st
import openai
from streamlit_chat import message
import streamlit_ext as ste
from docx import Document
from pypdf import PdfReader



#Python libraries
from typing import List
import re
import os
import mimetypes
from dotenv import load_dotenv, set_key, find_dotenv
import json


#langchain libraries
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import load_chain
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, QuestionAnswerPrompt



#Streanlit page config
st.set_page_config(page_title="AIDIA, AI Document Interactive Assistant!", page_icon="🤖", initial_sidebar_state="expanded")
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)



#Save the file in directory
@st.cache_data
def save_uploaded_file(uploadedfile):
    for file in uploadedfile:
        with open(os.path.join("tempDir", file.name), "wb") as f:
            f.write(file.getbuffer())
        st.success("Saved {} : To tempDir".format(file.name))

#Reading uploaded files and converting them into strings
@st.cache_data
def read_docx(directory_path: str) -> str:
    output = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.docx'):
            filepath = os.path.join(directory_path, filename)
            document = Document(filepath)
            for paragraph in document.paragraphs:
                text = paragraph.text
                # Merge hyphenated words
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                # Fix newlines in the middle of sentences
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                # Remove multiple newlines
                text = re.sub(r"\n\s*\n", "\n\n", text)
                output.append(text)
    return output

@st.cache_data
def read_txt(directory_path: str) -> str:
    output = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                # Merge hyphenated words
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                # Fix newlines in the middle of sentences
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                # Remove multiple newlines
                text = re.sub(r"\n\s*\n", "\n\n", text)
                output.append(text)
    return output

#Golden Function for pdf
def read_pdf(directory_path: str) -> str:
    output = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    # Merge hyphenated words
                    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                    # Fix newlines in the middle of sentences
                    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                    # Remove multiple newlines
                    text = re.sub(r"\n\s*\n", "\n\n", text)
                    output.append(text)
    return output


def format_transcript(data):
    transcript = json.loads(data)
    result = ""
    for item in transcript:
        if len(item) == 2:
            question, answer = item
            answer = answer.replace("\\n", "\n")
            result += f"{question}\n{answer}\n"
        else:
            st.write(f"Skipping item due to irregular structure: {item}")
    return result

def clear_files():
    # Delete any files in tempDir
    for filename in os.listdir(tempDir):
        file_path = os.path.join(tempDir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete index.json file
    index_path = os.path.join(os.getcwd(), "index.json")
    if os.path.exists(index_path):
        os.remove(index_path)

def create_temp_dir():
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")

#Indexing the uploaded files    
def construct_index(directory_path, api):
    openai.api_key = api
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.2, model_name="text-embedding-ada-002", max_tokens=num_outputs, openai_api_key = api))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index 


#Generate responses
def ask_ai(query_str):
    QA_PROMPT_TMPL = (
    "You are a chatbot assistant answering technical questions from a the {context_str}. If you do not know the answer to a question, or if the question is completely irrelevant to {context_str}, simply reply with: 'This doesn't seem to be related the index.' Given this information, please answer the question: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    # Build GPTSimpleVectorIndex
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query_str, response_mode="compact", text_qa_template=QA_PROMPT, mode="embedding")
    return response

def get_text(context, conversations):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=context,
            temperature=0.2,
            max_tokens=700,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    return response.choices[0].message.content




#Begin streamlit page
st.sidebar.header("About")
st.sidebar.markdown(
    """
This application was created by [Soroush Sabbaghan](mailto:ssabbagh@ucalgary.ca) using [Streamlit](https://streamlit.io/), [streamlit-chat](https://pypi.org/project/streamlit-chat/), [LangChain](https://github.com/hwchase17/langchain), and [OpenAI API](https://openai.com/api/)'s 
the most updated model [gpt-3.5-turbo](https://platform.openai.com/docs/models/overview) for educational purposes. 
"""
)

st.sidebar.header("Copyright")
st.sidebar.markdown(
    """
- This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
"""
)

st.title("Hi, I'm AIDIA 👋")
st.subheader("AI Document Interactive Assistant")
st.markdown("""___""")
st.subheader("1. Please Enter you OpenAI API key")
url = "https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key"
api = st.text_input("If you don't know your OpenAI API key click [here](%s)." % url, type="password", placeholder="Your API Key")

#API variables
env_path = find_dotenv()
if env_path == "":
    with open(".env", "w") as env_file:
        env_file.write("# .env\n")
        env_path = ".env"

# Load .env file
load_dotenv(dotenv_path=env_path)

set_key(env_path, "OPENAI_API_KEY", api)

openai.api_key = api
create_temp_dir()
tempDir = "tempDir/"

#API Check
if st.button("Check key"):
    if api is not None:
    #Delete any files in tempDir
        try:
            # Send a test request to the OpenAI API
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="What is the capital of France?",
                temperature=0.5
            )
            st.markdown("""---""")
            clear_files()
            st.success("API key is valid!")
        except Exception as e:
            st.error("API key is invalid: {}".format(e))
st.markdown("""---""") 

#Upload documents
st.subheader("2. Please Upload the files you wish to indexed as your knowledge base.")
docx_files = st.file_uploader("Upload Document", type=["pdf","docx","txt"], accept_multiple_files=True)
 
#Identifying file types and convert them into strings
if docx_files is not None:
    if st.button("Index Files!"):
        save_uploaded_file(docx_files)
        with st.spinner(text="Processing..."):
            
            for docx_files in os.listdir(tempDir):
                filepath = os.path.join(tempDir, docx_files)
                mime_type, _ = mimetypes.guess_type(filepath)

            if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = read_docx(tempDir)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_text(raw_text)
                embeddings = construct_index(tempDir, api)
                index = GPTSimpleVectorIndex.load_from_disk('index.json')
                st.subheader("Indexing complete! :white_check_mark:")
            elif mime_type == "text/plain":
                raw_text = read_txt(tempDir)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_text(raw_text)
                embeddings = construct_index(tempDir, api)
                index = GPTSimpleVectorIndex.load_from_disk('index.json')
                st.subheader("Indexing complete! :white_check_mark:")
            elif mime_type == "application/pdf":
                raw_text = read_pdf(tempDir)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_text(raw_text)
                embeddings = construct_index(tempDir, api)
                index = GPTSimpleVectorIndex.load_from_disk('index.json')
                st.subheader("Indexing complete! :white_check_mark:")
            else:
                st.write(f"Skipping file {filepath} with MIME type {mime_type}")

#Begin chatbot
st.markdown("""___""")
st.subheader("3. Start chatting with the bot by asking a question related to indexed document.")
if 'generated' not in st.session_state:
            st.session_state['generated'] = []

if 'past' not in st.session_state:
            st.session_state['past'] = []

#variables 
user_input =st.text_input("User:", key='input')
conversations = []

#Chat process
if st.button("Send"):
    with st.spinner("By code and algorithm, hear my call, let this AI speak with rhyme for all :magic_wand:"):
        
        response = ask_ai(user_input)
        output = str(response)
        #store the output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        #store all outputs
        conversations = [(st.session_state['past'][i], st.session_state["generated"][i]) for i in range(len(st.session_state['generated']))]

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

st.markdown("""___""")
if conversations:
   conversations_str = json.dumps(conversations)
   formatted_output = format_transcript(conversations_str)
   ste.download_button("Download Chat", formatted_output, "chat.txt")
