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

LLM_REPOSITORY='google/gemma-3-1b-it:Q8_0' # Isi hanya dengan Ollama. Ganti dengan repository LLM lainnya

GEMINI_API_KEY='' # Isi API KEY untuk penggunaan dengan Gemini
GEMINI_MODEL='gemini-2.5-flash-lite' # pilihan: gemini-2.5-flash, gemma-3-27b-it

VECTORSTORE_PATH='./storage/lpdp'
VECTORSTORE_INDEX='lpdp'

def chat_stream(prompt):
    for char in prompt:
        yield char 
        time.sleep(0.015)

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
        context = [txt for txt in f.readlines()]
        context = [remove_unwanted_char(txt) for txt in context]
        context = [txt for txt in context if txt != '' or len(txt) > 3] 
        
        print(context)
    
    context = ' '.join(context).replace('..', '.')
    with open(new_ctx_filename, 'a', encoding='utf-8') as f:
        f.write(context)
        f.close()
        
    return context     

def remove_unwanted_char(text: str):
    text = text.replace('\n', '').replace('\x0c', '')
    text = re.sub(r'^[\d\w]{0,2}+[\.\)]', '', text)
    text = re.sub(r'[\?\:\✓\*\●\;]', '', text)
    return text.strip()

def prepare_context():
    create_context(src_filename='public_policy_file_1751536393.pdf', 
                   dst_filename='lpdp-selected-page.pdf',
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
            
            response = generate(prompt, context) # generate_with_gemini(prompt, context)
            response = st.write_stream(chat_stream(response))
            
            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history)}",
                on_change=save_feedback,
                args=[len(st.session_state.history)],
            )
        st.session_state.history.append({"role": "assistant", "content": response})
            
def generate(question, ctx):
    question = prompt_cot(question, ctx)    
        
    print('\n', question, '\n')
        
    return request(question)

def prompt_cove(question, ctx):
    return f'''\nJawab pertanyaan dengan memberikan statement awal terhadap pertanyaan yang diajukan.
    Selelah statement dibuat, lakukan verifikasi terhadap pernyataan tersebut. 
    Gunakan [konteks] sebagai informasi tambahan untuk memperdalam pengetahuan dalam menjawab pertanyaan.
    Berikut adalah contoh dari proses verifikasi tersebut:

    [pertanyaan]: 
    Apakah Cristiano Ronaldo pemenang Euro 2004 ? 

    [statement]: 
    Cristiano Ronaldo adalah pemain muda Portugal yang menjuarai Final Euro 2004

    [verifikasi]:
    1. Ronaldo adalah benar pemain Portugal yang bermain pada Euro 2004
    2. Pada Euro 2004 Portugal adalah salah satu negara yang menjadi finalis
    3. Menemukan informasi bahwa pada Euro 2004 Portugal kalah dari Yunani
    4. Karena Portugal kalah dari Yunani maka juara Euro 2004 adalah Portugal, bukan Yunani
    5. Statement sebelumnya adalah benar karena Portugal kalah 1-0. Untuk itu Cristiano Ronaldo benar pemain muda Portugal, namun tidak menjuarai Final Euro 2004

    [jawaban]: Maka berdasarkan hasil verifikasi dapat dinyatakan bahwa Cristiano Ronaldo tidak menjuarai Euro 2004
    bersama Portugal. Tetapi benar bahwa Cristiano Ronaldo pemain muda Portugal pada kejuaran tersebut

    Jawablah pertanyaan berikut ini!
    
    [pertanyaan]: {question}
    [konteks]: {ctx}
    [jawaban]:'''

def prompt_cot(question, ctx):
    return f'''\nJawablah pertanyaan berikut dengan melakukan analisis secara sekuensial terhadap hal yang ditanyakan!
    Gunakan [konteks] sebagai informasi tambahan untuk memperdalam pengetahuan dalam menjawab pertanyaan.
    Berikut adalah contoh dari pertanyaan dan proses menjawab:

    [pertanyaan]: 
    Apakah Cristiano Ronaldo pemenang Euro 2004 ?

    [analisis]:
    1. Cristiano Ronaldo adalah pemain Portugal yang berposisi sebagai winger 
    2. Dia bermain bersama pemain top seperti Quaresma dan Deco 
    3. Portugal merupakan tuan rumah Euro 2004
    4. Pemenang Euro 2004 adalah tim yang menang pada partai final
    5. Portugal dikalahkan oleh Yunani dalam partai final dengan skor 1-0
    6. Gol tunggal dibuat oleh Angelos Charisteas

    [jawaban]: Cristiano Ronaldo bukanlah juara Euro 2024, melainkan Yunani yang menang dengan skor 1-0 melalui gol Angelos Charisteas
    
    Jawablah pertanyaan berikut ini!
    
    [pertanyaan]: {question}
    [konteks]: {ctx}
    [jawaban]:'''
    
    
def prompt_react(question, ctx):
    return f'''\nSelesaikan tugas menjawab pertanyaan dengan langkah-langkah yang dimasukkan: Pemikiran, Tindakan, dan Observasi. Langkah tersebut dapat dilakukan beberapa kali hingga jawaban akhir ditemukan. Pemikiran dapat merenungkan situasi saat ini, dan Tindakan dapat terdiri dari tiga jenis:  
        (1) Search[entitas], yang mencari Wikipedia untuk entitas yang sesuai dan mengembalikan paragraf pertama jika ada. Jika tidak ada, maka akan mengembalikan beberapa entitas serupa untuk dicari.  
        (2) Search[kata kunci], yang mengembalikan kalimat berikutnya yang mengandung kata kunci dalam bagian saat ini.  
        (3) Complete[jawaban], yang mengembalikan jawaban dan menyelesaikan tugas. 
        Konteks bersifat opsional. Gunakan konteks sebagai informasi hanya apabila relevan.
        
        Teks berikut adalah contoh langkah-langkah dalam menjawab pertanyaan:

        Pemikiran 1: Saya perlu mengetahui peran Cristiano Ronaldo dan kondisi tim Portugal pada Euro 2004.
        Tindakan 1: Search[informasi Ronaldo dan skuad Portugal Euro 2004]
        Observasi 1: Cristiano Ronaldo adalah winger Portugal yang bermain bersama pemain top seperti Quaresma dan Deco, dan Portugal menjadi tuan rumah Euro 2004.

        Pemikiran 2: Untuk menemukan pemenang Euro 2004, saya harus mengetahui hasil pertandingan final.
        Tindakan 2: Search[hasil final Euro 2004]
        Observasi 2: Portugal kalah dari Yunani di final Euro 2004 dengan skor 1–0.

        Pemikiran 3: Saya perlu mengetahui siapa pencetak gol dalam pertandingan final tersebut.
        Tindakan 3: Search[pencetak gol final Euro 2004]
        Observasi 3: Gol tunggal pertandingan final dicetak oleh Angelos Charisteas.
        
        Jawablah pertanyaan berikut ini dengan mengganti teks `<!jawaban>` dengan jawaban akhir!

        [pertanyaan]: {question}
        [konteks]: {ctx}
        
        <!jawaban>'''
    
def request(question):
    body = {
        'model': LLM_REPOSITORY,
        'messages': [
            { 
                'role': 'system', 
                'content': 'You are a helpful assistant.'
            },
            { 
                'role': 'user', 
                'content': question,
            }
        ],
        'max_tokens': 700
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
    response = re.sub(r'\s+', ' ', response)
    response = re.sub(r'(?<!^)(\d+\.)', r'\n\1', response)
    response = response.replace('[pertanyaan]:', '\n').replace('[statement]:', '\n').replace('[verifikasi]:', '').replace('[jawaban]:', '')
    return response
    
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
    results = vecstore.similarity_search_with_relevance_scores(query, 10)

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
    
    question = prompt_cot(question, ctx)    
    
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