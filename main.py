import os
import re
import sys
import json
import time
import requests
import faiss
import pymupdf 
import streamlit as st

from google import genai
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

LLM_REPOSITORY='' # Isi hanya dengan Ollama. Ganti dengan repository LLM lainnya

GEMINI_API_KEY='' # Isi API KEY untuk penggunaan dengan Gemini
GEMINI_MODEL='gemini-2.5-flash-lite' # pilihan: gemini-2.5-flash, gemma-3-27b-it

VECTORSTORE_PATH='./storage/lpdp'
VECTORSTORE_INDEX='lpdp'

def chat_stream(prompt):
    for char in prompt:
        yield char 
        time.sleep(0.02)

def save_feedback(index):
    st.session_state.history[index]['feedback'] = st.session_state[f'feedback_{index}']

def load_pdf(filename):
    if os.path.exists(filename):
        return pymupdf.open(filename)
    else:
        print(f'File tidak ditemukan')
        sys.exit(1)

def create_context(src_filename, dst_filename, ctx_filename):
    pdf = load_pdf(src_filename) 
    pdf.select([1,2,3,4,5,6,7,8,14,15])
    pdf.save(dst_filename)
    
    new_pdf = load_pdf(dst_filename)
    context = open(ctx_filename, 'wb')
    
    for page in tqdm(new_pdf, 'Parsing context...'):
        text = page.get_text().encode('utf-8')
        context.write(text)
        context.write(bytes((12,)))
    context.close()

def preprocess_context(ctx_filename, new_ctx_filename):
    if os.path.exists(ctx_filename) is False:
        print(f'File tidak ditemukan')
        sys.exit(1)
        
    with open(ctx_filename, 'r', encoding='utf-8') as f:
        context = [txt.strip() for txt in f.readlines() if txt not in ['\n', '\x0c', '']]
        context = [txt for txt in context if txt != '']
    
    context = ' '.join(context).replace('..', '.')
    with open(new_ctx_filename, 'a', encoding='utf-8') as f:
        f.write(context)
        f.close()
        
    return context     

def prepare_context():
    create_context(src_filename='public_policy_file_1751536393.pdf', 
                   dst_filename='lpdp-selection-page.pdf',
                   ctx_filename='context.txt')

    ctx = preprocess_context('context.txt', 'new_context.txt')
    store_context(ctx, VECTORSTORE_PATH, VECTORSTORE_INDEX)

def main():
    if "history" not in st.session_state:
        st.session_state.history = []

    for i, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                st.session_state[f"feedback_{i}"] = feedback
                st.feedback(
                    "thumbs",
                    key=f"feedback_{i}",
                    disabled=feedback is not None,
                    on_change=save_feedback,
                    args=[i],
                )

    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            context = get_context(prompt, VECTORSTORE_PATH, VECTORSTORE_INDEX)
            
            response = generate_with_gemini(prompt, context) # generate(prompt, context)
            response = st.write_stream(chat_stream(response))
            
            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history)}",
                on_change=save_feedback,
                args=[len(st.session_state.history)],
            )
        st.session_state.history.append({"role": "assistant", "content": response})
            
def generate(question, ctx):
    question = prompt(question, ctx)    
        
    print('\n', question, '\n')
        
    return request(question)
    
def prompt(question, ctx):
    return f'''\nJawablah pertanyaan berikut dan gunakan konteks berikut sebagai sumber informasi ! \npertanyaan: {question} \nkonteks: {ctx} \nJawaban: \n'''
    
def request(question):
    body = {
        'model': LLM_REPOSITORY,
        'messages': [
            { 
                'role': 'system', 
                'content': 'You are a helpful assistant'
            },
            { 
                'role': 'user', 
                'content': question,
            }
        ],
        'max_tokens': 500
    }
    
    response = requests.post(
        'http://localhost:11434/v1/chat/completions',
        headers = {
            'Content-Type': 'application/json'
        },
        data = json.dumps(body)
    )
    response = response.json()
    response = response['choices'][0]['message']['content'].replace('*', '').replace('\n', ' ')
    
    return re.sub(r'\s+', ' ', response) 

def faiss_store(embedding):
    index = faiss.IndexHNSWFlat(len(embedding.embed_query('helo')), 32)
    return FAISS(
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        embedding_function=embedding
    )

def faiss_client(storage_path, index_name, embedding):
    if os.path.exists(storage_path) is False:
        os.makedirs(storage_path)
        
    return FAISS.load_local(
        folder_path=storage_path,
        index_name=index_name,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

def get_context(query, path, index):
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', encode_kwargs={"normalize_embeddings": True})
    
    vecstore = faiss_client(path, index, embedding)
    results = vecstore.similarity_search_with_relevance_scores(query, 5)

    contexts = []
    for context in results:
        contexts.append(context[0].page_content)
        
    return ". ".join(contexts)
    
def store_context(ctx: str, path, index):
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', encode_kwargs={"normalize_embeddings": True})
    vecstore = faiss_store(embedding)
    
    documents = []
    contexts = ctx.split('.')
    for context in tqdm(contexts, 'Storing context...'):
        doc = Document(
            page_content=context,
        )
        documents.append(doc)
        
    vecstore.add_documents(
        documents=documents,
    )
    vecstore.save_local(folder_path=path, index_name=index)    
    
def generate_with_gemini(question, ctx):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    question = prompt(question, ctx)    
    
    print('\n', question, '\n')
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question
    )
    return response.text
    
if __name__ == '__main__':
    if 'prepare_context' not in st.session_state:
        prepare_context()
        st.session_state.prepare_context = True
        
    main()