# from fastapi import FastAPI, Query
# from pydantic import BaseModel
# from typing import List, Optional
# import re
# import sqlite3
# import pandas as pd
# from textblob import TextBlob
# import nltk
# from nltk.corpus import stopwords
# from collections import Counter
# import os

# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # --------- تحديد مسار قاعدة البيانات ---------
# DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

# # --------- إعداد الخدمة ---------
# app = FastAPI()
# stop_words = set(stopwords.words("english"))

# # --------- النماذج ---------
# class Options(BaseModel):
#     spelling_correction: bool = True
#     query_expansion: bool = True
#     query_suggestion: bool = True

# class QueryRequest(BaseModel):
#     query: str
#     options: Options

# class QueryResponse(BaseModel):
#     dataset: str
#     original_query: str
#     cleaned_query: str
#     corrected_query: Optional[str] = None
#     corrected_terms_diff: Optional[List[str]] = None
#     expanded_query: Optional[str] = None
#     expanded_terms_added: Optional[List[str]] = None
#     suggestions: Optional[List[str]] = None
#     refined_query: str

# # --------- المساعدات ---------
# def clean_input(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     tokens = text.split()
#     tokens = [t for t in tokens if t not in stop_words]
#     return " ".join(tokens)

# def correct_spelling(query: str) -> (str, List[str]):
#     words = query.split()
#     corrected_words = []
#     diffs = []
#     for word in words:
#         corrected = str(TextBlob(word).correct())
#         corrected_words.append(corrected)
#         if corrected != word:
#             diffs.append(f"{word} → {corrected}")
#     return " ".join(corrected_words), diffs

# def extract_synonyms_from_docs(dataset: str, query_words: List[str]) -> dict:
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     table_name = f"{dataset}_documents"  # مثل: trec_tot_documents
#     cursor.execute(f"SELECT clean_text FROM {table_name}")
#     docs = [row[0] for row in cursor.fetchall()]
#     conn.close()

#     all_words = []
#     for doc in docs:
#         all_words.extend(doc.split())

#     word_freq = Counter(all_words)
#     expansion_dict = {}

#     for word in query_words:
#         similar = [w for w in word_freq if w != word and w.startswith(word[:3])]
#         top_similar = sorted(similar, key=lambda w: -word_freq[w])[:3]
#         if top_similar:
#             expansion_dict[word] = top_similar

#     return expansion_dict

# def expand_query_with_docs(query: str, dataset: str) -> (str, List[str]):
#     words = query.split()
#     expansion_dict = extract_synonyms_from_docs(dataset, words)
#     expanded_words = []
#     added = []
#     for word in words:
#         if word in expansion_dict:
#             synonyms = expansion_dict[word]
#             expanded_words.append(f"({word} OR " + " OR ".join(synonyms) + ")")
#             added.append(f"{word} → " + ", ".join(synonyms))
#         else:
#             expanded_words.append(word)
#     return " ".join(expanded_words), added

# def suggest_queries(query: str) -> List[str]:
#     q = query.lower()
#     if q.startswith("how"):
#         return ["how to use embedding for retrieval?", "how to preprocess queries?"]
#     elif "cheap" in q:
#         return ["cheap laptops 2025", "best affordable phones"]
#     elif "movie" in q:
#         return ["top 10 movies 2025", "best action films"]
#     else:
#         return ["best information retrieval systems", "query expansion techniques"]

# def save_to_db(dataset: str, original: str, refined: str):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute(f"""
#         CREATE TABLE IF NOT EXISTS query_logs (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             dataset TEXT,
#             original_query TEXT,
#             refined_query TEXT
#         )
#     """)
#     cursor.execute(
#         "INSERT INTO query_logs (dataset, original_query, refined_query) VALUES (?, ?, ?)",
#         (dataset, original, refined)
#     )
#     conn.commit()
#     conn.close()

# # --------- نقطة النهاية ---------
# @app.post("/refine-query", response_model=QueryResponse)
# def refine_query(request: QueryRequest, dataset: str = Query("trec_tot")):
#     original = request.query
#     cleaned = clean_input(original)

#     corrected = cleaned
#     diffs = []
#     if request.options.spelling_correction:
#         corrected, diffs = correct_spelling(cleaned)

#     expanded = corrected
#     added = []
#     if request.options.query_expansion:
#         expanded, added = expand_query_with_docs(corrected, dataset)

#     suggestions = []
#     if request.options.query_suggestion:
#         suggestions = suggest_queries(corrected)

#     # حفظ الاستعلام
#     save_to_db(dataset, original, expanded)

#     return QueryResponse(
#         dataset=dataset,
#         original_query=original,
#         cleaned_query=cleaned,
#         corrected_query=corrected if corrected != cleaned else None,
#         corrected_terms_diff=diffs if diffs else None,
#         expanded_query=expanded if expanded != corrected else None,
#         expanded_terms_added=added if added else None,
#         suggestions=suggestions if suggestions else None,
#         refined_query=expanded
#     )
# from fastapi import FastAPI, Query
# from pydantic import BaseModel
# from typing import List, Optional
# import re
# import sqlite3
# import pandas as pd
# from textblob import TextBlob
# import nltk
# from nltk.corpus import stopwords
# from collections import Counter
# import os
# import joblib
# from sentence_transformers import util
# import torch

# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # --------- تحديد مسار قاعدة البيانات ---------
# DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

# # --------- إعداد الخدمة ---------
# app = FastAPI()
# stop_words = set(stopwords.words("english"))

# # --------- النماذج ---------
# class Options(BaseModel):
#     spelling_correction: bool = True
#     query_expansion: bool = True
#     query_suggestion: bool = True

# class QueryRequest(BaseModel):
#     query: str
#     options: Options

# class QueryResponse(BaseModel):
#     dataset: str
#     original_query: str
#     cleaned_query: str
#     corrected_query: Optional[str] = None
#     corrected_terms_diff: Optional[List[str]] = None
#     expanded_query: Optional[str] = None
#     expanded_terms_added: Optional[List[str]] = None
#     suggestions: Optional[List[str]] = None
#     refined_query: str

# # --------- تنظيف النص ---------
# def clean_input(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     tokens = text.split()
#     tokens = [t for t in tokens if t not in stop_words]
#     return " ".join(tokens)

# # --------- تصحيح الإملاء ---------
# def correct_spelling(query: str) -> (str, List[str]):
#     words = query.split()
#     corrected_words = []
#     diffs = []
#     for word in words:
#         corrected = str(TextBlob(word).correct())
#         corrected_words.append(corrected)
#         if corrected != word:
#             diffs.append(f"{word} → {corrected}")
#     return " ".join(corrected_words), diffs

# # --------- استخراج كلمات دلالية قريبة ---------
# def extract_semantic_expansion(dataset: str, query_words: List[str], top_k=3) -> dict:
   
#     # تحميل النموذج
 
#     MODEL_PATH = r"C:\Users\ASUS\Documents\ir_project2\offline_indexing_service\data\trec_tot\embedding_model\model.joblib"
#     model = joblib.load(MODEL_PATH)

#     # تحميل الوثائق من قاعدة البيانات
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute(f"SELECT clean_text FROM {dataset}_documents")
#     docs = [row[0] for row in cursor.fetchall()]
#     conn.close()

#     all_words = []
#     for doc in docs:
#         all_words.extend(doc.split())
#     word_freq = Counter(all_words)
#     common_words = [w for w, _ in word_freq.most_common(1000)]

#     word_embeddings = model.encode(common_words, convert_to_tensor=True)

#     expansion_dict = {}
#     for word in query_words:
#         word_embedding = model.encode(word, convert_to_tensor=True)
#         cosine_scores = util.cos_sim(word_embedding, word_embeddings)[0]
#         top_results = torch.topk(cosine_scores, k=top_k + 1)
#         similar = []
#         for idx in top_results.indices:
#             similar_word = common_words[int(idx)]
#             if similar_word != word:
#                 similar.append(similar_word)
#         if similar:
#             expansion_dict[word] = similar
#     return expansion_dict

# # --------- توسيع الاستعلام ---------
# def expand_query_with_docs(query: str, dataset: str) -> (str, List[str]):
#     words = query.split()
#     expansion_dict = extract_semantic_expansion(dataset, words)
#     expanded_words = []
#     added = []
#     for word in words:
#         if word in expansion_dict:
#             synonyms = expansion_dict[word]
#             expanded_words.append(f"({word} OR " + " OR ".join(synonyms) + ")")
#             added.append(f"{word} → " + ", ".join(synonyms))
#         else:
#             expanded_words.append(word)
#     return " ".join(expanded_words), added

# # --------- اقتراحات ذكية للاستعلام ---------
# def suggest_queries(query: str) -> List[str]:
#     q = query.lower()
#     if q.startswith("how"):
#         return ["how to use embedding for retrieval?", "how to preprocess queries?"]
#     elif "cheap" in q:
#         return ["cheap laptops 2025", "best affordable phones"]
#     elif "movie" in q:
#         return ["top 10 movies 2025", "best action films"]
#     else:
#         return ["best information retrieval systems", "query expansion techniques"]

# # --------- تسجيل الاستعلام في قاعدة البيانات ---------
# def save_to_db(dataset: str, original: str, refined: str):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute(f"""
#         CREATE TABLE IF NOT EXISTS query_logs (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             dataset TEXT,
#             original_query TEXT,
#             refined_query TEXT
#         )
#     """)
#     cursor.execute(
#         "INSERT INTO query_logs (dataset, original_query, refined_query) VALUES (?, ?, ?)",
#         (dataset, original, refined)
#     )
#     conn.commit()
#     conn.close()

# # --------- نقطة النهاية ---------
# @app.post("/refine-query", response_model=QueryResponse)
# def refine_query(request: QueryRequest, dataset: str = Query("trec_tot")):
#     original = request.query
#     cleaned = clean_input(original)

#     corrected = cleaned
#     diffs = []
#     if request.options.spelling_correction:
#         corrected, diffs = correct_spelling(cleaned)

#     expanded = corrected
#     added = []
#     if request.options.query_expansion:
#         expanded, added = expand_query_with_docs(corrected, dataset)

#     suggestions = []
#     if request.options.query_suggestion:
#         suggestions = suggest_queries(corrected)

#     # حفظ الاستعلام
#     save_to_db(dataset, original, expanded)

#     return QueryResponse(
#         dataset=dataset,
#         original_query=original,
#         cleaned_query=cleaned,
#         corrected_query=corrected if corrected != cleaned else None,
#         corrected_terms_diff=diffs if diffs else None,
#         expanded_query=expanded if expanded != corrected else None,
#         expanded_terms_added=added if added else None,
#         suggestions=suggestions if suggestions else None,
#         refined_query=expanded
#     )
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import re
import sqlite3
import os
import joblib
from collections import Counter
from sentence_transformers import util
import torch
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# --------- إعداد المسارات ---------
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
MODEL_PATH = r"C:\Users\ASUS\Documents\ir_project2\offline_indexing_service\data\trec_tot\embedding_model\model.joblib"

# --------- تحميل النماذج والبيانات global ---------
try:
    embedding_model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

stop_words = set(stopwords.words("english"))

# --------- FastAPI app ---------
app = FastAPI()

# --------- نماذج البيانات ---------
class Options(BaseModel):
    spelling_correction: bool = True
    query_expansion: bool = True
    query_suggestion: bool = True

class QueryRequest(BaseModel):
    query: str
    options: Options

class QueryResponse(BaseModel):
    dataset: str
    original_query: str
    cleaned_query: str
    corrected_query: Optional[str] = None
    corrected_terms_diff: Optional[List[str]] = None
    expanded_query: Optional[str] = None
    expanded_terms_added: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    refined_query: str

# --------- تنظيف النص ---------
def clean_input(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
# --------- تصحيح الإملاء ---------
def correct_spelling(query: str) -> (str, List[str]):
    """تصحيح إملاء الكلمات باستخدام TextBlob"""
    words = query.split()
    corrected_words = []
    diffs = []
    for word in words:
        corrected = str(TextBlob(word).correct())
        corrected_words.append(corrected)
        if corrected != word:
            diffs.append(f"{word} → {corrected}")
    return " ".join(corrected_words), diffs

# --------- تحميل الكلمات الشائعة offline (للتوسيع السريع) ---------
common_words_cache = {}

def load_common_words(dataset: str) -> List[str]:
    """تحميل الكلمات الشائعة من قاعدة البيانات والاحتفاظ بها في الذاكرة"""
    if dataset in common_words_cache:
        return common_words_cache[dataset]

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT clean_text FROM {dataset}_documents")
        docs = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error loading documents from DB: {e}")
        return []

    all_words = []
    for doc in docs:
        all_words.extend(doc.split())

    word_freq = Counter(all_words)
    common_words = [w for w, _ in word_freq.most_common(1000)]
    common_words_cache[dataset] = common_words
    return common_words

# --------- استخراج كلمات دلالية قريبة ---------
def extract_semantic_expansion(dataset: str, query_words: List[str], top_k=3) -> dict:
    """استخراج كلمات مشابهة دلاليًا لكل كلمة في الاستعلام"""
    if embedding_model is None:
        return {}

    common_words = load_common_words(dataset)
    if not common_words:
        return {}

    # حساب embeddings للكلمات الشائعة مرة واحدة فقط
    if f"{dataset}_word_embeddings" not in common_words_cache:
        try:
            embeddings = embedding_model.encode(common_words, convert_to_tensor=True)
            common_words_cache[f"{dataset}_word_embeddings"] = embeddings
        except Exception as e:
            print(f"Error encoding common words: {e}")
            return {}
    else:
        embeddings = common_words_cache[f"{dataset}_word_embeddings"]

    expansion_dict = {}
    for word in query_words:
        try:
            word_embedding = embedding_model.encode(word, convert_to_tensor=True)
            cosine_scores = util.cos_sim(word_embedding, embeddings)[0]
            top_results = torch.topk(cosine_scores, k=top_k + 1)
            similar = []
            for idx in top_results.indices:
                similar_word = common_words[int(idx)]
                if similar_word != word:
                    similar.append(similar_word)
            if similar:
                expansion_dict[word] = similar
        except Exception as e:
            print(f"Error in semantic expansion for '{word}': {e}")
            continue
    return expansion_dict

# --------- توسيع الاستعلام ---------
def expand_query_with_docs(query: str, dataset: str) -> (str, List[str]):
    """توسيع الاستعلام بالكلمات الدلالية المضافة"""
    words = query.split()
    expansion_dict = extract_semantic_expansion(dataset, words)
    expanded_words = []
    added = []
    for word in words:
        if word in expansion_dict:
            synonyms = expansion_dict[word]
            expanded_words.append(f"({word} OR " + " OR ".join(synonyms) + ")")
            added.append(f"{word} → " + ", ".join(synonyms))
        else:
            expanded_words.append(word)
    return " ".join(expanded_words), added

# --------- اقتراحات ذكية للاستعلام ---------
def suggest_queries(query: str) -> List[str]:
    """اقتراح استعلامات ذكية بسيطة بناءً على نص الاستعلام"""
    q = query.lower()
    if q.startswith("how"):
        return ["how to use embedding for retrieval?", "how to preprocess queries?"]
    elif "cheap" in q:
        return ["cheap laptops 2025", "best affordable phones"]
    elif "movie" in q:
        return ["top 10 movies 2025", "best action films"]
    else:
        return ["best information retrieval systems", "query expansion techniques"]

# --------- تسجيل الاستعلام في قاعدة البيانات ---------
def save_to_db(dataset: str, original: str, refined: str):
    """حفظ سجل الاستعلام الأصلي والاستعلام المحسن في قاعدة البيانات"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT,
                original_query TEXT,
                refined_query TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO query_logs (dataset, original_query, refined_query) VALUES (?, ?, ?)",
            (dataset, original, refined)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving query to DB: {e}")

# --------- نقطة النهاية ---------
@app.post("/refine-query", response_model=QueryResponse)
def refine_query(request: QueryRequest, dataset: str = Query("trec_tot")):
    """
    استقبال استعلام، وتنظيفه، وتصحيح إملائه، وتوسيعه، واقتراح استعلامات بديلة،
    ثم حفظ النتيجة وإرجاعها.
    """
    original = request.query
    cleaned = clean_input(original)

    corrected = cleaned
    diffs = []
    if request.options.spelling_correction:
        corrected, diffs = correct_spelling(cleaned)

    expanded = corrected
    added = []
    if request.options.query_expansion:
        expanded, added = expand_query_with_docs(corrected, dataset)

    suggestions = []
    if request.options.query_suggestion:
        suggestions = suggest_queries(corrected)

    save_to_db(dataset, original, expanded)

    return QueryResponse(
        dataset=dataset,
        original_query=original,
        cleaned_query=cleaned,
        corrected_query=corrected if corrected != cleaned else None,
        corrected_terms_diff=diffs if diffs else None,
        expanded_query=expanded if expanded != corrected else None,
        expanded_terms_added=added if added else None,
        suggestions=suggestions if suggestions else None,
        refined_query=expanded
    )
