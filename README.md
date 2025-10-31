# RAG PDF + Streamlit

Aplikasi Retrieval-Augmented Generation sederhana dengan context yang diambil dari konten dalam file pdf.
Dalam repository ini mengambil study case panduan LPDP.

Note: dalam menggunakan teknologi generatif apapun pastikan untuk melakukan verifikasi terhadap hasil.
apapun yang dalam repository ini adalah untuk keperluan demo.

## Fitur

- **PyMuPDF** digunakan untuk mengekstrak konten dari pdf. Dalam demo ini hanya beberapa halaman yang diseleksi sebagai candidate context
- Konteks disimpan dalam vector database **FAISS**
- Model embedding yang digunakan yaitu **BAAI/bge-m3**
- Dapat menggunakan **Ollama** atau **Gemini API** sebagai backend inference untuk LLM
- Chat menggunakan **Streamlit**

## Instalasi

Buat **virtual environment** terlebih dahulu.

```sh
python.exe -m venv env
```

Aktifkan virtual environment nya. Jalan lupa sesuaikan dengan OS yang digunakan.
Dalam demo ini menggunakan Windows

```sh
.\env\Script\Activate
```

Kemudian install depedensinya (package)

```sh
pip install -r requirements.txt
```

Jalankan streamlit

```sh
streamlit run main.py
```

## Modifikasi

Jika menggunakan Ollama, jangan lupa mengisi repo di main.py 
**LLM_REPOSITORY="<nama-repository>"**

Jika menggunakan Gemini, lengkapi
**GEMINI_API_KEY="<api-key>"**
**GEMINI_MODEL="<model-gemini>"**


gunakan function **generate_with_gemini()** untuk Gemini atau **generate()** untuk Ollama

## Penting

Hasil dari setiap LLM akan berbeda. Silahkan sesuaikan prompt pada function **prompt()** agar instruksi lebih baik.
Mungkin **Chain-of-Thought (CoT)** atau **Chain-of-Verification (CoVe)** bisa menjadi solusi