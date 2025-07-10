import sqlite3
from fastapi import FastAPI, Query
from collections import defaultdict
import json
import os
from pathlib import Path
import joblib
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

DATABASE_PATH =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

@app.get("/build-inverted-index")
def build_inverted_index(
    dataset: str = Query("trec_tot"),
    text_column: str = Query("clean_text"),
    output_file: str = Query("inverted2_index.json")
):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    documents_table = f"{dataset}_documents"

    try:
        cursor.execute(f"SELECT doc_id, {text_column} FROM {documents_table}")
        rows = cursor.fetchall()
    except Exception as e:
        return {"error": f"❌ خطأ في الوصول إلى جدول الوثائق: {e}"}

    index = defaultdict(set)

    for doc_id, clean_text in rows:
        if not clean_text:
            continue
        tokens = clean_text.split()  # النص منظف مسبقاً، فقط تقسيم
        for term in tokens:
            index[term].add(str(doc_id))

    # تحويل الـ set إلى list من أجل JSON
    index = {term: list(doc_ids) for term, doc_ids in index.items()}

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
    output_path = os.path.join(base_path, output_file)

    # تأكد أن المجلد موجود
    os.makedirs(base_path, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    return {
        "✅": "تم بناء فهرس معكوس Inverted2 Index",
        "terms": len(index),
        "saved_to": output_path
    }


@app.get("/search-tfidf-db")
def search_tfidf_from_db(
    query_id: str = Query(...),
    dataset: str = Query("trec_tot"),  # أو "antique"
    index_file: str = Query("inverted2_index.json"),
    tfidf_matrix_file: str = Query("tfidf_docs_matrix.npz"),
    tfidf_model_file: str = Query("tfidf_vectorizer.pkl"),
    doc_ids_file: str = Query("tfidf_doc_ids.json"),
    top_k: int = Query(10)
):
    # تحديد المسارات حسب مجموعة البيانات
    base_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            f"../offline_indexing_service/data/{dataset}"
        )
    )
    fallback_path = str(Path("C:/Users/ASUS/Documents/ir_project2/data") / dataset)

    index_path = os.path.join(fallback_path, index_file)
    tfidf_matrix_path = os.path.join(base_path, tfidf_matrix_file)
    tfidf_model_path = os.path.join(base_path, tfidf_model_file)
    doc_ids_path = os.path.join(fallback_path, doc_ids_file)

    # التحقق من وجود الملفات المطلوبة
    for file_path in [index_path, tfidf_matrix_path, tfidf_model_path, doc_ids_path]:
        if not os.path.exists(file_path):
            return {"error": f"❌ الملف غير موجود: {file_path}"}

    # تحميل الفهرس المعكوس وأسماء الوثائق
    with open(index_path, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)

    with open(doc_ids_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    # تحميل مصفوفة TF-IDF والنموذج
    tfidf_matrix = scipy.sparse.load_npz(tfidf_matrix_path)
    tfidf_vectorizer = joblib.load(tfidf_model_path)

    # الاتصال بقاعدة البيانات وجلب الاستعلام المنظف
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        query_table = f"{dataset}_queries"
        cursor.execute(f"SELECT clean_text FROM {query_table} WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        return {"error": f"❌ خطأ أثناء الوصول إلى قاعدة البيانات: {e}"}

    if not row:
        return {"error": f"❌ لم يتم العثور على استعلام بهذا المعرف: {query_id}"}

    clean_query = row[0]
    if not clean_query:
        return {"error": "❌ الاستعلام موجود لكن النص المنظف فارغ"}

    # معالجة الاستعلام
    tokens = clean_query.split()
    if not tokens:
        return {"error": "❌ الاستعلام المنظف لا يحتوي على كلمات"}

    # تحديد الوثائق المرشحة عبر الفهرس المعكوس
    candidate_docs = set()
    for term in tokens:
        candidate_docs.update(inverted_index.get(term, []))

    if not candidate_docs:
        return {"query_id": query_id, "message": "❌ لم يتم العثور على وثائق مرشحة للاستعلام"}

    # تصفية الوثائق المرشحة في المصفوفة
    candidate_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id in candidate_docs]
    if not candidate_indices:
        return {"query_id": query_id, "error": "❌ لم يتم العثور على وثائق بعد الفلترة من الفهرس"}

    candidate_matrix = tfidf_matrix[candidate_indices]
    candidate_doc_ids = [doc_ids[i] for i in candidate_indices]

    # تمثيل الاستعلام
    query_vector = tfidf_vectorizer.transform([" ".join(tokens)])

    # حساب التشابه بين الاستعلام والوثائق المرشحة
    similarities = cosine_similarity(query_vector, candidate_matrix).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = [
        {
            "doc_id": candidate_doc_ids[idx],
            "score": float(similarities[idx])
        }
        for idx in top_indices
    ]

    return {
        "query_id": query_id,
        "clean_query": clean_query,
        "top_k": top_k,
        "candidates_considered": len(candidate_doc_ids),
        "results": results
    }


