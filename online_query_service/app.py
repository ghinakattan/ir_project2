from fastapi import FastAPI
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
from fastapi import FastAPI, Query, HTTPException
import numpy as np

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
    query: str           # الاستعلام الأصلي (غير منظف)
    dataset: str = "trec_tot"
    top_k: int = 10

# ================================
# تحميل الموارد عند بدء التشغيل
# ================================
@app.on_event("startup")
def load_resources():
    dataset = "trec_tot"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    tfidf_vectorizer_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_vectorizer.pkl')
    tfidf_matrix_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'tfidf_docs_matrix.npz')
    doc_ids_path = os.path.join(base_dir, 'data', dataset, 'tfidf_doc_ids.json')

    assert os.path.exists(tfidf_vectorizer_path), f"❌ الملف غير موجود: {tfidf_vectorizer_path}"
    assert os.path.exists(tfidf_matrix_path), f"❌ الملف غير موجود: {tfidf_matrix_path}"
    assert os.path.exists(doc_ids_path), f"❌ الملف غير موجود: {doc_ids_path}"

    app.state.vectorizer = joblib.load(tfidf_vectorizer_path)
    app.state.tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    with open(doc_ids_path, "r") as f:
        app.state.doc_ids = json.load(f)

    print("✅ تم تحميل الموارد بنجاح.")

# ================================
# خدمة المطابقة (TF-IDF) مع تنظيف داخلي - الطلب الخامس
# ================================
@app.post("/query-match")
def query_match(request: MatchRequest):
    # تنظيف النص داخل الدالة
    cleaned_query = clean_text(request.query)

    vectorizer = app.state.vectorizer
    tfidf_matrix = app.state.tfidf_matrix
    doc_ids = app.state.doc_ids

    query_vec = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    top_indices = scores.argsort()[::-1][:request.top_k]
    results = [{"doc_id": doc_ids[i], "score": float(scores[i])} for i in top_indices]

    return {
        "query": request.query,            # النص الأصلي
        "cleaned_query": cleaned_query,   # النص بعد التنظيف
        "top_k": request.top_k,
        "results": results
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
    dataset = request.dataset
    top_k = request.top_k

    # تنظيف الاستعلام
    cleaned_query = clean_text(request.query)

    # TF-IDF (من الحالة)
    vectorizer = app.state.vectorizer
    tfidf_matrix = app.state.tfidf_matrix
    doc_ids_tfidf = app.state.doc_ids  # doc_ids المخزنة في الحالة

    query_vec_tfidf = vectorizer.transform([cleaned_query])
    scores_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix)[0]
    tfidf_scores_dict = dict(zip(doc_ids_tfidf, scores_tfidf))

    # Embedding (نفس طريقة query_embedding)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'embedding_model', 'model.joblib')
    doc_embeddings_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'doc_embeddings.npy')
    doc_ids_path = os.path.join(base_dir, 'offline_indexing_service', 'data', dataset, 'embedding_doc_ids.json')

    if not os.path.exists(model_path) or not os.path.exists(doc_embeddings_path) or not os.path.exists(doc_ids_path):
        raise HTTPException(status_code=404, detail="❌ ملفات Embedding غير موجودة")

    model = joblib.load(model_path)
    doc_embeddings = np.load(doc_embeddings_path)
    with open(doc_ids_path, "r") as f:
        doc_ids_emb = json.load(f)
    if not isinstance(doc_ids_emb, list):
        raise HTTPException(status_code=500, detail="❌ محتوى embedding_doc_ids.json غير صحيح")

    query_vec_emb = model.encode([cleaned_query])
    scores_emb = cosine_similarity(query_vec_emb, doc_embeddings)[0]
    emb_scores_dict = dict(zip(doc_ids_emb, scores_emb))

    # دمج نتائج TF-IDF و Embedding
    all_doc_ids = set(doc_ids_tfidf).union(set(doc_ids_emb))
    hybrid_scores = []
    for doc_id in all_doc_ids:
        score_tfidf = tfidf_scores_dict.get(doc_id, 0)
        score_emb = emb_scores_dict.get(doc_id, 0)
        avg_score = (score_tfidf + score_emb) / 2
        hybrid_scores.append((doc_id, avg_score))

    # ترتيب واختيار أعلى النتائج
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = hybrid_scores[:top_k]
    results = [{"doc_id": doc_id, "score": float(score)} for doc_id, score in top_results]

    return {
        "query": request.query,
        "cleaned_query": cleaned_query,
        "top_k": top_k,
        "results": results
    }
