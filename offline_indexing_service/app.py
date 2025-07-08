from fastapi import FastAPI, Query, APIRouter
import os
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import csv
from sentence_transformers import util
from pathlib import Path

app = FastAPI()


DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
 
def save_results_to_db(df_results, table_name, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            query_id TEXT,
            doc_id TEXT,
            score REAL,
            rank INTEGER
        )
    """)
    c.execute(f"DELETE FROM {table_name}")
    records = df_results.to_records(index=False)
    c.executemany(f"INSERT INTO {table_name} (query_id, doc_id, score, rank) VALUES (?, ?, ?, ?)", records)
    conn.commit()
    conn.close()


def get_table_name(dataset):
    if dataset == "trec_tot":
        return "trec_tot_documents"
    elif dataset == "antique":
        return "antique_documents"
    else:
        raise ValueError("❌ اسم الداتا سيت غير مدعوم")


# BASE_DIR الآن يشير إلى: offline_indexing_service/data
BASE_DIR = Path(__file__).resolve().parent / "data"


# ------------- TF-IDF Representation -------------
@app.get("/build-tfidf")
def build_tfidf(
    dataset: str = "trec_tot",
    model_name: str = "tfidf_vectorizer.pkl",
    matrix_name: str = "tfidf_docs_matrix.npz"
):
    try:
        table_name = get_table_name(dataset)

        # حساب المسار الصحيح لقاعدة البيانات
        base_dir = os.path.dirname(__file__)
        absolute_db_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'ir_docs.db'))

        # الاتصال بقاعدة البيانات
        conn = sqlite3.connect(absolute_db_path)
        cursor = conn.cursor()

        # استخراج النصوص المنظفة
        cursor.execute(f"SELECT clean_text FROM {table_name}")
        cleaned_texts = [row[0] for row in cursor.fetchall() if row[0] is not None]
        conn.close()

        if not cleaned_texts:
            return {"error": "❌ لا توجد نصوص منظفة في قاعدة البيانات."}

        vectorizer = TfidfVectorizer(
            lowercase=False,
            preprocessor=None,
            tokenizer=None,
            token_pattern=r"(?u)\b\w+\b"
        )

        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

        # حفظ النموذج والمصفوفة
        output_dir = os.path.join("data", dataset)
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, model_name)
        matrix_path = os.path.join(output_dir, matrix_name)

        joblib.dump(vectorizer, model_path)
        sparse.save_npz(matrix_path, tfidf_matrix)

        return {
            "✅": "تم بناء نموذج TF-IDF بنجاح",
            "dataset": dataset,
            "model_file": model_path,
            "matrix_file": matrix_path,
            "documents": tfidf_matrix.shape[0],
            "features": tfidf_matrix.shape[1]
        }

    except Exception as e:
        return {"error": f"❌ حصل خطأ: {str(e)}"}
   
# ------------- Embedding Representation -------------
@app.get("/build-embeddings")
def build_embeddings(
    dataset: str = Query(..., description="اسم مجموعة البيانات مثل 'trec_tot' أو 'antique'"),
    model_name: str = Query("all-MiniLM-L6-v2", description="اسم نموذج SentenceTransformer")
):
    table_name = f"{dataset}_documents"

    try:
        # ✅ قراءة النصوص من قاعدة البيانات
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT doc_id, clean_text FROM {table_name}", conn)
        conn.close()

        if "clean_text" not in df.columns or df.empty:
            return {"error": "❌ لا توجد نصوص منظفة في قاعدة البيانات"}

        texts = df["clean_text"].astype(str).tolist()

        # ✅ تحميل نموذج التمثيل
        model = SentenceTransformer(model_name)

        # ✅ استخراج التمثيلات الشعاعية
        embeddings = model.encode(texts, show_progress_bar=True)

        # ✅ مسارات الحفظ
        base_path = os.path.join("data", dataset)
        os.makedirs(base_path, exist_ok=True)

        model_dir = os.path.join(base_path, "embedding_model")
        embeddings_path = os.path.join(base_path, "doc_embeddings.npy")

        # ✅ حفظ النموذج والتمثيلات
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        np.save(embeddings_path, embeddings)

        return {
            "✅": "تم بناء التمثيل الشعاعي بنجاح",
            "dataset": dataset,
            "model_used": model_name,
            "num_documents": len(texts),
            "embedding_dim": embeddings.shape[1],
            "saved_model": model_dir,
            "saved_embeddings": embeddings_path
        }

    except Exception as e:
        return {"error": f"❌ حصل خطأ: {str(e)}"}

# ------------- Hybrid Representation -------------
@app.get("/save-hybrid-run")
def save_hybrid_run(
    dataset: str = Query(..., description="اسم مجموعة البيانات مثل 'trec_tot' أو 'antique'"),
    tfidf_file: str = Query("tfidf_run_all.csv"),
    embedding_file: str = Query("embedding_run_all.csv"),
    output_file: str = Query("hybrid_run_all.csv"),
    run_name: str = Query("hybrid_avg")
):
    base_output = os.path.join(BASE_DIR, dataset)
    tfidf_path = os.path.join(base_output, tfidf_file)
    emb_path = os.path.join(base_output, embedding_file)
    output_path = os.path.join(base_output, output_file)

    if not os.path.exists(tfidf_path) or not os.path.exists(emb_path):
        return {"error": "❌ تأكد من وجود ملفات TF-IDF و Embedding run"}

    df_tfidf = pd.read_csv(tfidf_path)
    df_emb = pd.read_csv(emb_path)

    df_tfidf["query_id"] = df_tfidf["query_id"].astype(str)
    df_tfidf["doc_id"] = df_tfidf["doc_id"].astype(str)
    df_emb["query_id"] = df_emb["query_id"].astype(str)
    df_emb["doc_id"] = df_emb["doc_id"].astype(str)

    merged = pd.merge(
        df_tfidf,
        df_emb,
        on=["query_id", "doc_id"],
        how="outer",
        suffixes=('_tfidf', '_emb')
    )

    merged["score_tfidf"] = merged["score_tfidf"].fillna(0)
    merged["score_emb"] = merged["score_emb"].fillna(0)

    merged["score"] = (merged["score_tfidf"] + merged["score_emb"]) / 2

    hybrid_run = merged[["query_id", "doc_id", "score"]]

    # التعيينات تتم قبل الreturn
    hybrid_run["rank"] = hybrid_run.groupby("query_id")["score"].rank(method="first", ascending=False)
    hybrid_run["run_name"] = run_name

    hybrid_run = hybrid_run.sort_values(by=["query_id", "rank"])

    hybrid_run.to_csv(output_path, index=False)

    table_name = f"{dataset}_hybrid_results"
    save_results_to_db(hybrid_run[["query_id", "doc_id", "score", "rank"]], table_name)

    # جهز أمثلة للرجوع بها في الresponse
    example_query = hybrid_run['query_id'].iloc[0]
    example_docs = hybrid_run[hybrid_run['query_id'] == example_query].head(3).to_dict(orient='records')

    return {
        "✅": "تم دمج السكورات (TF-IDF + Embedding) وحفظ النتائج",
        "saved_file": output_path,
        "saved_db_table": table_name,
        "total_pairs": len(hybrid_run),
        "unique_queries": hybrid_run['query_id'].nunique(),
        "example_query": example_query,
        "example_docs": example_docs
    }

@app.get("/save-tfidf-run")
def save_tfidf_run(
    dataset: str = Query(..., description="اسم مجموعة البيانات مثل 'trec_tot' أو 'antique'"),
    vectorizer_file: str = Query("tfidf_vectorizer.pkl"),
    matrix_file: str = Query("tfidf_docs_matrix.npz"),
    output_file: str = Query("tfidf_run_all.csv"),
    top_k: int = Query(10),
    save_all: bool = Query(False)
):
    base_output = os.path.join(BASE_DIR, dataset)
    vectorizer_path = os.path.join(base_output, vectorizer_file)
    matrix_path = os.path.join(base_output, matrix_file)
    output_path = os.path.join(base_output, output_file)

    if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path]):
        return {"error": "❌ ملفات التمثيل غير موجودة."}

    conn = sqlite3.connect(DB_PATH)
    df_docs = pd.read_sql_query(f"SELECT * FROM {dataset}_documents", conn)
    df_queries = pd.read_sql_query(f"SELECT * FROM {dataset}_queries", conn)
    conn.close()

    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = sparse.load_npz(matrix_path)

    doc_ids = df_docs["doc_id"].astype(str).tolist()
    all_results = []

    for _, row in df_queries.iterrows():
        query_id = str(row["query_id"])
        query_text = str(row["clean_text"])  # نص الاستعلام المنظف
        query_vec = vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

        if save_all:
            top_indices = np.arange(len(similarities))
        else:
            top_indices = similarities.argsort()[::-1][:top_k]

        for idx in top_indices:
            all_results.append({
                "query_id": query_id,
                "doc_id": doc_ids[idx],
                "score": float(similarities[idx])
            })

    df_results = pd.DataFrame(all_results)
    df_results["rank"] = df_results.groupby("query_id")["score"].rank(method="first", ascending=False)

    # حفظ CSV
    df_results.to_csv(output_path, index=False)

    # حفظ في قاعدة البيانات
    table_name = f"{dataset}_tfidf_results"
    save_results_to_db(df_results, table_name)

    return {
        "✅": "تم تنفيذ البحث باستخدام TF-IDF وحفظ النتائج",
        "saved_file": output_path,
        "saved_db_table": table_name,
        "queries": len(df_queries),
        "documents": len(df_docs),
        "saved_results": len(df_results),
        "top_k": top_k,
        "save_all": save_all
    }

@app.get("/save-embedding-run")
def save_embedding_run(
    dataset: str = Query(..., description="اسم مجموعة البيانات مثل 'trec_tot' أو 'antique'"),
    model_file: str = Query("embedding_model/model.joblib"),
    output_file: str = Query("embedding_run_all.csv"),
    top_k: int = Query(10),
    save_all: bool = Query(False)
):
    base_output = os.path.join(BASE_DIR, dataset)
    model_path = os.path.join(base_output, model_file)
    output_path = os.path.join(base_output, output_file)

    if not os.path.exists(model_path):
        return {"error": f"❌ ملف النموذج غير موجود: {model_path}"}

    conn = sqlite3.connect(DB_PATH)
    df_docs = pd.read_sql_query(f"SELECT * FROM {dataset}_documents", conn)
    df_queries = pd.read_sql_query(f"SELECT * FROM {dataset}_queries", conn)
    conn.close()

    model = joblib.load(model_path)
    docs_ids = df_docs["doc_id"].astype(str).tolist()
    docs_texts = df_docs["clean_text"].astype(str).tolist()
    docs_embeddings = model.encode(docs_texts, convert_to_tensor=True, show_progress_bar=True)

    all_results = []

    for _, row in df_queries.iterrows():
        query_id = str(row["query_id"])
        query_text = str(row["clean_text"])
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, docs_embeddings)[0]

        if save_all:
            top_indices = np.arange(len(cosine_scores))
        else:
            top_indices = np.argpartition(-cosine_scores, range(top_k))[:top_k]

        for idx in top_indices:
            all_results.append({
                "query_id": query_id,
                "doc_id": docs_ids[idx],
                "score": float(cosine_scores[idx])
            })

    df_results = pd.DataFrame(all_results)
    df_results["rank"] = df_results.groupby("query_id")["score"].rank(method="first", ascending=False)

    # حفظ CSV
    df_results.to_csv(output_path, index=False)

    # حفظ في قاعدة البيانات
    table_name = f"{dataset}_embedding_results"
    save_results_to_db(df_results, table_name)

    return {
        "✅": "تم تنفيذ البحث باستخدام Embedding وحفظ النتائج",
        "saved_file": output_path,
        "saved_db_table": table_name,
        "queries": len(df_queries),
        "documents": len(df_docs),
        "saved_results": len(df_results),
        "top_k": top_k,
        "save_all": save_all
    }



