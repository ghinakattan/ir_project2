from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import json
import joblib
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
import time
from sentence_transformers import SentenceTransformer
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
class MatchRequest(BaseModel):
    query: str
    dataset: str = "trec_tot"
    top_k: int = 10

# ================================
# خدمة TF-IDF
# ================================
@app.post("/query_match")
def query_match(request: MatchRequest):
    start_time = time.time()

    dataset = request.dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    tfidf_vectorizer_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_vectorizer.pkl')
    tfidf_matrix_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_docs_matrix.npz')
    doc_ids_path = os.path.join(base_dir, 'data', dataset, 'tfidf_doc_ids.json')

    if not os.path.exists(tfidf_vectorizer_path) or not os.path.exists(tfidf_matrix_path) or not os.path.exists(doc_ids_path):
        raise HTTPException(status_code=404, detail="ملفات TF-IDF غير موجودة للمجموعة المحددة")

    vectorizer = joblib.load(tfidf_vectorizer_path)
    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    with open(doc_ids_path, "r") as f:
        doc_ids = json.load(f)

    cleaned_query = clean_text(request.query)
    query_vec = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    top_indices = scores.argsort()[::-1][:request.top_k]
    results = [{"doc_id": doc_ids[i], "score": float(scores[i])} for i in top_indices]

    duration = round(time.time() - start_time, 4)

    return {
        "query": request.query,
        "cleaned_query": cleaned_query,
        "dataset": dataset,
        "top_k": request.top_k,
        "results": results,
        "duration_seconds": duration
    }



@app.post("/query-embedding")
def query_embedding(request: MatchRequest):
    dataset = request.dataset
    top_k = request.top_k

    print(f"Dataset requested: {dataset}")

    # تنظيف الاستعلام
    cleaned_query = clean_text(request.query)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'embedding_model', 'model.joblib')
    doc_embeddings_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'doc_embeddings.npy')
    doc_ids_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'embedding_doc_ids.json')

    print(f"Model path: {model_path}")
    print(f"Doc embeddings path: {doc_embeddings_path}")
    print(f"Doc IDs path: {doc_ids_path}")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"❌ model.joblib غير موجود: {model_path}")
    if not os.path.exists(doc_embeddings_path):
        raise HTTPException(status_code=404, detail=f"❌ doc_embeddings.npy غير موجود: {doc_embeddings_path}")
    if not os.path.exists(doc_ids_path):
        raise HTTPException(status_code=404, detail=f"❌ embedding_doc_ids.json غير موجود: {doc_ids_path}")

    model = joblib.load(model_path)
    doc_embeddings = np.load(doc_embeddings_path)

    with open(doc_ids_path, "r") as f:
        doc_ids = json.load(f)

    # تحقق من أن doc_ids قائمة
    if not isinstance(doc_ids, list):
        raise HTTPException(status_code=500, detail=f"❌ محتوى embedding_doc_ids.json غير صحيح (ليس قائمة) في: {doc_ids_path}")

    query_vec = model.encode([cleaned_query])
    scores = cosine_similarity(query_vec, doc_embeddings)[0]

    top_indices = scores.argsort()[::-1][:top_k]
    results = [{"doc_id": doc_ids[i], "score": float(scores[i])} for i in top_indices]

    return {
        "query": request.query,
        "cleaned_query": cleaned_query,
        "top_k": top_k,
        "results": results
    }
@app.post("/query-hybrid")
def query_hybrid(request: MatchRequest):
    start_time = time.time()

    dataset = request.dataset
    top_k = request.top_k
    cleaned_query = clean_text(request.query)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # مسارات ملفات TF-IDF
    tfidf_vectorizer_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_vectorizer.pkl')
    tfidf_matrix_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_docs_matrix.npz')
    tfidf_doc_ids_path = os.path.join(base_dir, 'data', dataset, 'tfidf_doc_ids.json')

    # مسارات ملفات Embedding
    model_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'embedding_model', 'model.joblib')
    doc_embeddings_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'doc_embeddings.npy')
    embedding_doc_ids_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'embedding_doc_ids.json')

    # تحقق من وجود ملفات TF-IDF
    if not os.path.exists(tfidf_vectorizer_path) or not os.path.exists(tfidf_matrix_path) or not os.path.exists(tfidf_doc_ids_path):
        raise HTTPException(status_code=404, detail="ملفات TF-IDF غير موجودة للمجموعة المحددة")

    # تحقق من وجود ملفات Embedding
    if not os.path.exists(model_path) or not os.path.exists(doc_embeddings_path) or not os.path.exists(embedding_doc_ids_path):
        raise HTTPException(status_code=404, detail="ملفات Embedding غير موجودة للمجموعة المحددة")

    # تحميل موارد TF-IDF
    vectorizer = joblib.load(tfidf_vectorizer_path)
    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    with open(tfidf_doc_ids_path, "r") as f:
        doc_ids_tfidf = json.load(f)

    # تحميل موارد Embedding
    model = joblib.load(model_path)
    doc_embeddings = np.load(doc_embeddings_path)
    with open(embedding_doc_ids_path, "r") as f:
        doc_ids_emb = json.load(f)

    if not isinstance(doc_ids_tfidf, list) or not isinstance(doc_ids_emb, list):
        raise HTTPException(status_code=500, detail="محتوى ملفات معرّفات الوثائق غير صحيح")

    # حساب نتائج TF-IDF
    query_vec_tfidf = vectorizer.transform([cleaned_query])
    scores_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix)[0]
    tfidf_scores_dict = dict(zip(doc_ids_tfidf, scores_tfidf))

    # حساب نتائج Embedding
    query_vec_emb = model.encode([cleaned_query])
    scores_emb = cosine_similarity(query_vec_emb, doc_embeddings)[0]
    emb_scores_dict = dict(zip(doc_ids_emb, scores_emb))

    # دمج النتائج (متوسط درجات TF-IDF و Embedding)
    all_doc_ids = set(doc_ids_tfidf).union(set(doc_ids_emb))
    hybrid_scores = []
    for doc_id in all_doc_ids:
        score_tfidf = tfidf_scores_dict.get(doc_id, 0)
        score_emb = emb_scores_dict.get(doc_id, 0)
        avg_score = (score_tfidf + score_emb) / 2
        hybrid_scores.append((doc_id, avg_score))

    # ترتيب النتائج واختيار الأعلى
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = hybrid_scores[:top_k]
    results = [{"doc_id": doc_id, "score": float(score)} for doc_id, score in top_results]

    duration = round(time.time() - start_time, 4)

    return {
        "query": request.query,
        "cleaned_query": cleaned_query,
        "dataset": dataset,
        "top_k": top_k,
        "results": results,
        "duration_seconds": duration
    }
    

@app.post("/query-match-clustered")
def query_match_clustered(request: MatchRequest):
    start_time = time.perf_counter()

    dataset = request.dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    try:
        # مسارات الملفات
        vectorizer_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_vectorizer.pkl')
        matrix_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_docs_matrix.npz')
        doc_ids_path = os.path.join(base_dir, 'data', dataset, 'tfidf_doc_ids.json')
        clusters_csv_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_clusters.csv')

        # التحقق من وجود الملفات
        for path in [vectorizer_path, matrix_path, doc_ids_path, clusters_csv_path]:
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail=f"ملف {os.path.basename(path)} غير موجود")

        # تحميل الملفات
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = sparse.load_npz(matrix_path)
        with open(doc_ids_path, "r") as f:
            doc_ids = json.load(f)
        df_clusters = pd.read_csv(clusters_csv_path)
        df_clusters['doc_id'] = df_clusters['doc_id'].astype(str)
        doc_cluster_map = dict(zip(df_clusters['doc_id'], df_clusters['cluster_label']))
        docid_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        # تجهيز الاستعلام
        cleaned_query = clean_text(request.query)
        query_vec = vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vec, tfidf_matrix)[0]

        # حساب مجموع الدرجات حسب الكلستر
        clusters_scores = {}
        for doc_id, score in zip(doc_ids, scores):
            cluster_label = doc_cluster_map.get(doc_id, -1)
            clusters_scores.setdefault(cluster_label, 0)
            clusters_scores[cluster_label] += score

        sorted_clusters = sorted(clusters_scores.items(), key=lambda x: x[1], reverse=True)

        # ترتيب الوثائق داخل كل كلستر
        results = []
        for cluster_label, _ in sorted_clusters:
            cluster_docs = [doc_id for doc_id, cl_label in doc_cluster_map.items() if cl_label == cluster_label]
            cluster_docs_scores = []
            for doc_id in cluster_docs:
                idx = docid_to_index.get(doc_id)
                if idx is not None:
                    cluster_docs_scores.append((doc_id, scores[idx]))
            cluster_docs_scores.sort(key=lambda x: x[1], reverse=True)
            results.extend(cluster_docs_scores)

        results = results[:request.top_k]

        execution_time = round(time.perf_counter() - start_time, 4)

        return {
            "query": request.query,
            "cleaned_query": cleaned_query,
            "dataset": dataset,
            "top_k": request.top_k,
            "duration_seconds": execution_time,
            "results": [
                {"doc_id": doc_id, "score": float(score)} for doc_id, score in results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

