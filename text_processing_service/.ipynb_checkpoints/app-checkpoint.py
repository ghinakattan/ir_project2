from fastapi import FastAPI, Query
import pandas as pd
import os
import re
import nltk
import sqlite3
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

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


# ✅ المسار الصحيح لقاعدة البيانات داخل مجلد data
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

@app.get("/load-docs-to-db")
def load_docs_to_db(
    dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (trec_tot أو antique)"),
    file_name: str = Query(..., description="📄 اسم ملف CSV (مثلاً trec_tot_docs.csv)")
):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
    csv_path = os.path.join(base_path, file_name)

    if not os.path.exists(csv_path):
        return {"error": f"❌ الملف غير موجود: {csv_path}"}

    df = pd.read_csv(csv_path)
    if "doc_id" not in df.columns or "doc_text" not in df.columns:
        return {"error": "❌ الملف يجب أن يحتوي على الأعمدة: doc_id, doc_text"}

    df["clean_text"] = ""  # عمود فارغ مبدئياً
    table_name = f"{dataset}_documents"

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    return {
        "✅": "تم تحميل البيانات بنجاح إلى قاعدة البيانات",
        "dataset": dataset,
        "table": table_name,
        "documents": len(df)
    }


@app.get("/clean-docs-in-db")
def clean_docs_in_db(
    dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (trec_tot أو antique)")
):
    table_name = f"{dataset}_documents"

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    if "doc_text" not in df.columns or "doc_id" not in df.columns:
        conn.close()
        return {"error": f"❌ الجدول لا يحتوي على الأعمدة المطلوبة: doc_id, doc_text"}

    df["clean_text"] = df["doc_text"].astype(str).apply(clean_text)

    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    # حفظ نسخة CSV تلقائيًا بعد التنظيف
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
    os.makedirs(base_path, exist_ok=True)
    clean_csv_path = os.path.join(base_path, f"{dataset}_docs_clean.csv")
    df.to_csv(clean_csv_path, index=False)

    return {
        "✅": "تم تنظيف النصوص وتحديث قاعدة البيانات وحفظ نسخة CSV",
        "dataset": dataset,
        "table": table_name,
        "documents_cleaned": len(df),
        "clean_csv_path": clean_csv_path
    }


@app.get("/save-docs-to-db")
def save_docs_to_db(
    dataset: str = Query("trec_tot", description="اسم مجموعة البيانات: trec_tot أو antique"),
    file_name: str = Query(None, description="📄 اسم ملف CSV المنظف (افتراضي حسب مجموعة البيانات)")
):
    # تعيين اسم الملف المنظف حسب المجموعة
    if dataset == "trec_tot":
        file_name = file_name or "trec_tot_docs_clean.csv"
        table_name = "trec_tot_documents"
    elif dataset == "antique":
        file_name = file_name or "antique_docs_clean.csv"
        table_name = "antique_documents"
    else:
        return {"error": f"❌ اسم مجموعة البيانات غير مدعوم: {dataset}"}

    # تحديد المسار الكامل للملف
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
    file_path = os.path.join(base_path, file_name)

    if not os.path.exists(file_path):
        return {"error": f"❌ الملف غير موجود: {file_path}"}

    # قراءة البيانات
    df = pd.read_csv(file_path)

    # التحقق من الأعمدة المطلوبة
    if 'doc_id' not in df.columns or 'doc_text' not in df.columns or 'clean_text' not in df.columns:
        return {"error": "❌ الملف يجب أن يحتوي على الأعمدة: doc_id, doc_text, clean_text"}

    # حفظ إلى قاعدة البيانات
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        return {"error": f"❌ حدث خطأ أثناء الحفظ: {str(e)}"}

    return {
        "✅": "تم حفظ المستندات إلى قاعدة البيانات بنجاح",
        "dataset": dataset,
        "table_name": table_name,
        "stored_documents": len(df),
        "db_path": DB_PATH
    }


@app.get("/show-docs-from-db")
def show_docs_from_db(
    dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (trec_tot أو antique)"),
    limit: int = Query(10, description="عدد الوثائق التي تريد عرضها")
):
    table_name = f"{dataset}_documents"

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
    conn.close()

    return {
        "dataset": dataset,
        "shown_documents": len(df),
        "documents": df.to_dict(orient="records")
    }

@app.get("/clean-queries")
def clean_queries(
    dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (trec_tot أو antique)"),
    file_name: str = Query(None, description="📄 اسم ملف الاستعلامات (افتراضي حسب مجموعة البيانات)")
):
    import sqlite3

    # 📁 ملف الاستعلامات يتم قراءته من:
    # Documents/ir_project2/data/{dataset}/{dataset}_queries.csv
    if file_name is None:
        file_name = f"{dataset}_queries.csv"

    # 🗂️ المسار الكامل إلى مجلد مجموعة البيانات
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))

    # 📄 المسار الكامل لملف الاستعلامات الأصلي
    file_path = os.path.join(base_path, file_name)

    # ❌ التحقق من وجود الملف
    if not os.path.exists(file_path):
        return {"error": f"❌ ملف الاستعلامات غير موجود: {file_path}"}

    # 📄 قراءة ملف الاستعلامات
    df = pd.read_csv(file_path)

    # ✅ التحقق من وجود الأعمدة الأساسية
    if "query_id" not in df.columns or "query_text" not in df.columns:
        return {"error": "❌ الملف يجب أن يحتوي على الأعمدة: query_id, query_text"}

    # 🧹 تنظيف عمود الاستعلامات
    df["clean_text"] = df["query_text"].astype(str).apply(clean_text)

    # 💾 حفظ الاستعلامات المنظفة في ملف جديد:
    # Documents/ir_project2/data/{dataset}/{dataset}_queries_clean.csv
    cleaned_file_path = os.path.join(base_path, f"{dataset}_queries_clean.csv")
    df.to_csv(cleaned_file_path, index=False)

    query_table = f"{dataset}_queries"

    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql(query_table, conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        return {
            "✅": "تم تنظيف الاستعلامات وحفظها في ملف، لكن حصل خطأ أثناء الحفظ في قاعدة البيانات",
            "cleaned_file": cleaned_file_path,
            "error": str(e)
        }

    return {
        "✅": "تم تنظيف الاستعلامات بنجاح، وحفظها كملف وكجدول في قاعدة البيانات",
        "original_file": file_path,
        "cleaned_file": cleaned_file_path,
        "db_table": query_table,
        "queries_cleaned": len(df)
    }
