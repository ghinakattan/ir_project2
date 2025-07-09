# from fastapi import FastAPI
# from pydantic import BaseModel
# import os
# import joblib
# import pandas as pd
# import numpy as np
# from scipy import sparse
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer, PorterStemmer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# app = FastAPI()

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()

# def custom_preprocessor(text):
#     if text is None:
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text

# def custom_tokenizer(text):
#     text = custom_preprocessor(text)
#     tokens = text.split()
#     tokens = [t for t in tokens if t not in stop_words]
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     tokens = [stemmer.stem(t) for t in tokens]
#     return tokens

# # تحميل البيانات مرة واحدة عند بدء التطبيق
# DATASETS = {}

# def load_resources(dataset="trec_tot"):
#     if dataset in DATASETS:
#         return DATASETS[dataset]

#     base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))

#     vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")
#     matrix_path = os.path.join(base_path, "tfidf_docs_matrix.npz")
#     docs_file = "tot_docs_clean.csv" if dataset == "trec_tot" else f"{dataset}_docs_clean.csv"
#     docs_path = os.path.join(base_path, docs_file)

#     if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path, docs_path]):
#         raise FileNotFoundError("❌ أحد الملفات المطلوبة غير موجود")

#     vectorizer = joblib.load(vectorizer_path)
#     tfidf_matrix = sparse.load_npz(matrix_path)
#     df_docs = pd.read_csv(docs_path)

#     DATASETS[dataset] = {
#         "vectorizer": vectorizer,
#         "tfidf_matrix": tfidf_matrix,
#         "df_docs": df_docs
#     }

#     return DATASETS[dataset]

# # نموذج الطلب
# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 10


# @app.post("/query-match")
# def query_match(request: QueryRequest, dataset: str = "trec_tot"):
#     try:
#         resources = load_resources(dataset)
#     except FileNotFoundError as e:
#         return {"error": str(e)}

#     vectorizer = resources["vectorizer"]
#     tfidf_matrix = resources["tfidf_matrix"]
#     df = resources["df_docs"]

#     query_tokens = custom_tokenizer(request.query)
#     query_preprocessed = " ".join(query_tokens)

#     query_vec = vectorizer.transform([query_preprocessed])
#     similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
#     top_indices = similarities.argsort()[::-1][:request.top_k]
#     top_scores = similarities[top_indices]

#     results = []
#     for idx, score in zip(top_indices, top_scores):
#         results.append({
#             "doc_id": str(df.iloc[idx]["doc_id"]),
#             "score": float(score),
#             "text": df.iloc[idx]["doc_text"]
#         })

#     return {
#         "query": request.query,
#         "cleaned_query": query_preprocessed,
#         "top_k": request.top_k,
#         "results": results
#     }


# # ----- استعلام Embedding مع تنظيف الاستعلام -----
# @app.post("/query-embedding")
# def query_embedding(request: QueryRequest,
#                     dataset: str = "trec_tot",
#                     embedding_model_name: str = "all-MiniLM-L6-v2",
#                     embeddings_file: str = "doc_embeddings.npy",
#                     # source: str = "trec_tot_docs.csv",
#                     text_column: str = "doc_text",
#                     id_column: str = "doc_id"):
#     source = "tot_docs_clean.csv" if dataset == "trec_tot" else f"{dataset}_docs_clean.csv"

#     BASE_DIR = get_base_dir(dataset)
#     embeddings_path = os.path.join(BASE_DIR, embeddings_file)
#     source_path = os.path.join(BASE_DIR, source)

#     if not all(os.path.exists(p) for p in [embeddings_path, source_path]):
#         return {"error": "❌ ملف التمثيلات أو الوثائق غير موجود"}

#     df = pd.read_csv(source_path)
#     embeddings = np.load(embeddings_path)

#     model = SentenceTransformer(embedding_model_name)

#     query_tokens = custom_tokenizer(request.query)
#     query_preprocessed = " ".join(query_tokens)
#     print(f"Original Query: {request.query}")
#     print(f"Cleaned Tokens: {query_tokens}")
#     print(f"Preprocessed Query String: {query_preprocessed}")

#     query_embedding_vec = model.encode([query_preprocessed])[0]
#     similarities = cosine_similarity([query_embedding_vec], embeddings).flatten()
#     print("Similarities with first 5 documents:", similarities[:5])

#     top_indices = similarities.argsort()[::-1][:request.top_k]
#     top_scores = similarities[top_indices]

#     results = []
#     for idx, score in zip(top_indices, top_scores):
#         doc_info = {
#             "doc_id": str(df.iloc[idx][id_column]) if id_column in df.columns else idx,
#             "score": float(score),
#             "text": df.iloc[idx][text_column]
#         }
#         results.append(doc_info)

#     return {
#         "query": request.query,
#         "cleaned_query": query_preprocessed,
#         "top_k": request.top_k,
#         "results": results
#     }

# # ----- استعلام Hybrid دمج TF-IDF و Embedding (التوزيعي Parallel) -----
# @app.post("/query-hybrid")
# def query_hybrid(request: QueryRequest,
#                  dataset: str = "trec_tot",
#                  vectorizer_name: str = "tfidf_vectorizer.pkl",
#                  matrix_name: str = "tfidf_docs_matrix.npz",
#                  embeddings_file: str = "doc_embeddings.npy",
#                  embedding_model_name: str = "all-MiniLM-L6-v2",
#                  # source: str = "trec_tot_docs.csv",
#                  text_column: str = "doc_text",
#                  id_column: str = "doc_id",
#                  alpha: float = 0.5  # وزن دمج النتيجتين (0-1)
#                  ):
#     source = "tot_docs_clean.csv" if dataset == "trec_tot" else f"{dataset}_docs_clean.csv"

#     BASE_DIR = get_base_dir(dataset)
#     vectorizer_path = os.path.join(BASE_DIR, vectorizer_name)
#     matrix_path = os.path.join(BASE_DIR, matrix_name)
#     embeddings_path = os.path.join(BASE_DIR, embeddings_file)
#     source_path = os.path.join(BASE_DIR, source)

#     if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path, embeddings_path, source_path]):
#         return {"error": "❌ أحد الملفات المطلوبة غير موجود"}

#     vectorizer = joblib.load(vectorizer_path)
#     tfidf_matrix = sparse.load_npz(matrix_path)
#     embeddings = np.load(embeddings_path)
#     df = pd.read_csv(source_path)
#     model = SentenceTransformer(embedding_model_name)

#     query_tokens = custom_tokenizer(request.query)
#     query_preprocessed = " ".join(query_tokens)
#     print(f"Original Query: {request.query}")
#     print(f"Cleaned Tokens: {query_tokens}")
#     print(f"Preprocessed Query String: {query_preprocessed}")

#     query_vec_tfidf = vectorizer.transform([query_preprocessed])
#     query_embedding_vec = model.encode([query_preprocessed])[0]

#     sim_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix).flatten()
#     sim_emb = cosine_similarity([query_embedding_vec], embeddings).flatten()
#     print("TF-IDF similarities first 5 docs:", sim_tfidf[:5])
#     print("Embedding similarities first 5 docs:", sim_emb[:5])

#     sim_hybrid = alpha * sim_tfidf + (1 - alpha) * sim_emb

#     top_indices = sim_hybrid.argsort()[::-1][:request.top_k]
#     top_scores = sim_hybrid[top_indices]

#     results = []
#     for idx, score in zip(top_indices, top_scores):
#         doc_info = {
#             "doc_id": str(df.iloc[idx][id_column]) if id_column in df.columns else idx,
#             "score": float(score),
#             "text": df.iloc[idx][text_column]
#         }
#         results.append(doc_info)

#     return {
#         "query": request.query,
#         "cleaned_query": query_preprocessed,
#         "top_k": request.top_k,
#         "alpha": alpha,
#         "results": results
#     }

from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

app = FastAPI()

# ================================
# إعدادات المعالجة النصية
# ================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# ================================
# نماذج البيانات
# ================================
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    cleaned_query: str

class MatchRequest(BaseModel):
    cleaned_query: str
    dataset: str = "trec_tot"
    top_k: int = 10

# ================================
# خدمة تنظيف الاستعلام (الطلب الرابع)
# ================================
@app.post("/clean-query", response_model=QueryResponse)
def clean_query(request: QueryRequest):
    cleaned = clean_text(request.query)
    return {"cleaned_query": cleaned}

# ================================
# تحميل الموارد عند بدء التشغيل
# ================================
@app.on_event("startup")
def load_resources():
    dataset = "trec_tot"
    # احسب المسار إلى مجلد ir_project2 بشكل صحيح:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    print("DEBUG: base_dir =", base_dir)

    tfidf_vectorizer_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_vectorizer.pkl')
    tfidf_matrix_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_docs_matrix.npz')
    doc_ids_path = os.path.join(base_dir, 'data', dataset, 'tfidf_doc_ids.json')

    print("DEBUG: tfidf_vectorizer_path =", tfidf_vectorizer_path)
    print("DEBUG: tfidf_matrix_path =", tfidf_matrix_path)
    print("DEBUG: doc_ids_path =", doc_ids_path)
    print("DEBUG: doc_ids_path exists?", os.path.exists(doc_ids_path))

    assert os.path.exists(tfidf_vectorizer_path), f"❌ الملف غير موجود: {tfidf_vectorizer_path}"
    assert os.path.exists(tfidf_matrix_path), f"❌ الملف غير موجود: {tfidf_matrix_path}"
    assert os.path.exists(doc_ids_path), f"❌ الملف غير موجود: {doc_ids_path}"

    app.state.vectorizer = joblib.load(tfidf_vectorizer_path)
    app.state.tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    with open(doc_ids_path, "r") as f:
        app.state.doc_ids = json.load(f)

    print("✅ تم تحميل الموارد بنجاح.")




# ================================
# خدمة المطابقة (TF-IDF) - الطلب الخامس
# ================================
@app.post("/query-match")
def query_match(request: MatchRequest):
    vectorizer = app.state.vectorizer
    tfidf_matrix = app.state.tfidf_matrix
    doc_ids = app.state.doc_ids

    query_vec = vectorizer.transform([request.cleaned_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    top_indices = scores.argsort()[::-1][:request.top_k]
    results = [{"doc_id": doc_ids[i], "score": float(scores[i])} for i in top_indices]

    return {
        "query": request.cleaned_query,
        "top_k": request.top_k,
        "results": results
    }
