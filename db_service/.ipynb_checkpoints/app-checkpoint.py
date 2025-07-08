# from fastapi import FastAPI, Query
# import pandas as pd
# import os
# import sqlite3
# from typing import List

# app = FastAPI()

# @app.get("/insert-docs-to-db")
# def insert_docs_to_db(
#     dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (مثلاً trec_tot أو antique)"),
#     docs_file: str = Query(None, description="اسم ملف الوثائق المنظفة (مثلاً tot_docs_clean.csv)")
# ):
#     # تعيين الملف المناسب إذا لم يتم تمريره
#     if docs_file is None:
#         docs_file = f"{dataset}_docs_clean.csv"

#     base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
#     docs_path = os.path.join(base_path, docs_file)

#     if not os.path.exists(docs_path):
#         return {"error": f"❌ ملف الوثائق غير موجود: {docs_path}"}
#     if not os.path.exists(db_path):
#         return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

#     df = pd.read_csv(docs_path)
#     if "doc_id" not in df.columns or "clean_text" not in df.columns:
#         return {"error": "❌ الملف لا يحتوي على الأعمدة المطلوبة: doc_id, clean_text"}

#     df["dataset"] = dataset
#     df["doc_text"] = df["doc_text"].fillna("")
#     df["clean_text"] = df["clean_text"].fillna("")

#     # الاتصال بقاعدة البيانات
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     inserted = 0
#     for _, row in df.iterrows():
#         try:
#             cursor.execute("""
#                 INSERT OR REPLACE INTO documents (doc_id, dataset, doc_text, clean_text)
#                 VALUES (?, ?, ?, ?)
#             """, (str(row["doc_id"]), dataset, row["doc_text"], row["clean_text"]))
#             inserted += 1
#         except Exception as e:
#             continue

#     conn.commit()
#     conn.close()

#     return {
#         "✅": "تم إدخال الوثائق إلى قاعدة البيانات",
#         "dataset": dataset,
#         "file": docs_file,
#         "inserted": inserted
#     }

# @app.get("/get-doc-from-db")
# def get_doc_from_db(
#     doc_id: str = Query(..., description="معرّف الوثيقة"),
#     dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (مثلاً trec_tot أو antique)")
# ):
#     # مسار قاعدة البيانات
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

#     if not os.path.exists(db_path):
#         return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

#     # الاتصال بقاعدة البيانات
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # تنفيذ الاستعلام
#     cursor.execute("""
#         SELECT doc_id, dataset, doc_text, clean_text 
#         FROM documents 
#         WHERE doc_id = ? AND dataset = ?
#     """, (doc_id, dataset))

#     row = cursor.fetchone()
#     conn.close()

#     if row is None:
#         return {"error": f"❌ لم يتم العثور على الوثيقة: doc_id={doc_id}, dataset={dataset}"}

#     return {
#         "doc_id": row[0],
#         "dataset": row[1],
#         "doc_text": row[2],
#         "clean_text": row[3]
#     }

# @app.get("/list-docs-from-db")
# def list_docs_from_db(
#     dataset: str = Query("trec_tot", description="اسم مجموعة البيانات (trec_tot أو antique)"),
#     limit: int = Query(10, description="عدد الوثائق المراد عرضها")
# ):
#     # مسار قاعدة البيانات
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

#     if not os.path.exists(db_path):
#         return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

#     # الاتصال وتنفيذ الاستعلام
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     cursor.execute("""
#         SELECT doc_id, dataset, doc_text, clean_text 
#         FROM documents 
#         WHERE dataset = ?
#         LIMIT ?
#     """, (dataset, limit))

#     rows = cursor.fetchall()
#     conn.close()

#     docs = []
#     for row in rows:
#         docs.append({
#             "doc_id": row[0],
#             "dataset": row[1],
#             "doc_text": row[2],
#             "clean_text": row[3]
#         })

#     return {
#         "dataset": dataset,
#         "documents_returned": len(docs),
#         "documents": docs
#     }

# @app.get("/list-db-tables")
# def list_db_tables():
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

#     if not os.path.exists(db_path):
#         return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     tables = cursor.fetchall()
#     conn.close()

#     return {
#         "tables": [t[0] for t in tables]
#     }
# @app.get("/show-table-content")
# def show_table_content(
#     table_name: str = Query(..., description="اسم الجدول داخل قاعدة البيانات"),
#     limit: int = Query(10, description="عدد الصفوف التي سيتم عرضها")
# ):
#     # تحديد مسار قاعدة البيانات
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_documents.db'))

#     if not os.path.exists(db_path):
#         return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

#     try:
#         conn = sqlite3.connect(db_path)
#         df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
#         conn.close()

#         return {
#             "table": table_name,
#             "rows_returned": len(df),
#             "data": df.to_dict(orient="records")
#         }

#     except Exception as e:
#         return {"error": str(e)}
# /////////////////////////////////////////////////////////////////////////////////////////////////
# from fastapi import FastAPI, Query
# import pandas as pd
# import os
# import sqlite3

# app = FastAPI()

# # ✅ إدخال الوثائق إلى جدول مخصص لكل مجموعة بيانات
# @app.get("/insert-docs-to-db")
# def insert_docs_to_db(
#     dataset: str = Query(..., description="مثلاً trec_tot أو antique"),
#     docs_file: str = Query(None, description="مثلاً trec_tot_docs_clean.csv أو antique_docs_clean.csv")
# ):
#     dataset = dataset.strip()  # ✅ إصلاح الفراغات الزائدة

#     if docs_file is None:
#         docs_file = f"{dataset}_docs_clean.csv"

#     base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
#     docs_path = os.path.join(base_path, docs_file)

#     if not os.path.exists(docs_path):
#         return {"error": f"❌ ملف الوثائق غير موجود: {docs_path}"}
#     if not os.path.exists(db_path):
#         return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

#     df = pd.read_csv(docs_path)
#     if "doc_id" not in df.columns or "clean_text" not in df.columns:
#         return {"error": "❌ الملف لا يحتوي على الأعمدة المطلوبة: doc_id و clean_text"}

#     df["doc_text"] = df["doc_text"].fillna("")
#     df["clean_text"] = df["clean_text"].fillna("")

#     table_name = f"{dataset}_documents"

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # ✅ إنشاء الجدول إذا لم يكن موجوداً
#     cursor.execute(f"""
#         CREATE TABLE IF NOT EXISTS {table_name} (
#             doc_id TEXT PRIMARY KEY,
#             doc_text TEXT,
#             clean_text TEXT
#         )
#     """)

#     inserted = 0
#     for _, row in df.iterrows():
#         try:
#             cursor.execute(f"""
#                 INSERT OR REPLACE INTO {table_name} (doc_id, doc_text, clean_text)
#                 VALUES (?, ?, ?)
#             """, (str(row["doc_id"]), row["doc_text"], row["clean_text"]))
#             inserted += 1
#         except Exception:
#             continue

#     conn.commit()
#     conn.close()

#     return {
#         "✅": f"تم إدخال {inserted} وثيقة إلى جدول {table_name}",
#         "dataset": dataset,
#         "file": docs_file
#     }

# # ✅ استرجاع وثيقة من جدول مخصص للمجموعة
# @app.get("/get-doc-from-db")
# def get_doc_from_db(
#     doc_id: str = Query(...),
#     dataset: str = Query(...),
# ):
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

#     table_name = f"{dataset}_documents"

#     if not os.path.exists(db_path):
#         return {"error": "❌ قاعدة البيانات غير موجودة"}

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     cursor.execute(f"""
#         SELECT doc_id, doc_text, clean_text FROM {table_name}
#         WHERE doc_id = ?
#     """, (doc_id,))

#     row = cursor.fetchone()
#     conn.close()

#     if not row:
#         return {"error": f"❌ لم يتم العثور على الوثيقة في {table_name}"}

#     return {
#         "doc_id": row[0],
#         "doc_text": row[1],
#         "clean_text": row[2]
#     }

# # ✅ عرض مجموعة من الوثائق من جدول محدد
# @app.get("/list-docs-from-db")
# def list_docs_from_db(
#     dataset: str = Query(...),
#     limit: int = Query(10)
# ):
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
#     table_name = f"{dataset}_documents"

#     if not os.path.exists(db_path):
#         return {"error": "❌ قاعدة البيانات غير موجودة"}

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     cursor.execute(f"""
#         SELECT doc_id, doc_text, clean_text FROM {table_name}
#         LIMIT ?
#     """, (limit,))

#     rows = cursor.fetchall()
#     conn.close()

#     return {
#         "dataset": dataset,
#         "documents": [
#             {"doc_id": row[0], "doc_text": row[1], "clean_text": row[2]}
#             for row in rows
#         ]
#     }

# # ✅ عرض جميع الجداول داخل قاعدة البيانات
# @app.get("/list-db-tables")
# def list_db_tables():
#     db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

#     if not os.path.exists(db_path):
#         return {"error": "❌ قاعدة البيانات غير موجودة"}

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     tables = cursor.fetchall()
#     conn.close()

#     return {
#         "tables": [t[0] for t in tables]
#     }

from fastapi import FastAPI, Query
import pandas as pd
import os
import sqlite3

app = FastAPI()

# ✅ دالة مساعدة لتنظيف اسم مجموعة البيانات
def clean_dataset_name(name: str) -> str:
    return name.strip()

# ✅ إدخال الوثائق إلى جدول مخصص لكل مجموعة بيانات
@app.get("/insert-docs-to-db")
def insert_docs_to_db(
    dataset: str = Query(..., description="مثلاً trec_tot أو antique"),
    docs_file: str = Query(None, description="مثلاً tot_docs_clean.csv أو antique_docs_clean.csv")
):
    dataset = clean_dataset_name(dataset)

    if docs_file is None:
        docs_file = f"{dataset}_docs_clean.csv"

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
    docs_path = os.path.join(base_path, docs_file)

    if not os.path.exists(docs_path):
        return {"error": f"❌ ملف الوثائق غير موجود: {docs_path}"}
    if not os.path.exists(db_path):
        return {"error": f"❌ قاعدة البيانات غير موجودة: {db_path}"}

    df = pd.read_csv(docs_path)
    if "doc_id" not in df.columns or "clean_text" not in df.columns:
        return {"error": "❌ الملف لا يحتوي على الأعمدة المطلوبة: doc_id و clean_text"}

    df["doc_text"] = df["doc_text"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")

    table_name = f"{dataset}_documents"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ✅ إنشاء الجدول إذا لم يكن موجوداً
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            doc_id TEXT PRIMARY KEY,
            doc_text TEXT,
            clean_text TEXT
        )
    """)

    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name} (doc_id, doc_text, clean_text)
                VALUES (?, ?, ?)
            """, (str(row["doc_id"]), row["doc_text"], row["clean_text"]))
            inserted += 1
        except Exception:
            continue

    conn.commit()
    conn.close()

    return {
        "✅": f"تم إدخال {inserted} وثيقة إلى جدول {table_name}",
        "dataset": dataset,
        "file": docs_file
    }

# ✅ استرجاع وثيقة من جدول مخصص للمجموعة
@app.get("/get-doc-from-db")
def get_doc_from_db(
    doc_id: str = Query(...),
    dataset: str = Query(...)
):
    dataset = clean_dataset_name(dataset)
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
    table_name = f"{dataset}_documents"

    if not os.path.exists(db_path):
        return {"error": "❌ قاعدة البيانات غير موجودة"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT doc_id, doc_text, clean_text FROM {table_name}
        WHERE doc_id = ?
    """, (doc_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"error": f"❌ لم يتم العثور على الوثيقة في {table_name}"}

    return {
        "doc_id": row[0],
        "doc_text": row[1],
        "clean_text": row[2]
    }

# ✅ عرض مجموعة من الوثائق من جدول محدد
@app.get("/list-docs-from-db")
def list_docs_from_db(
    dataset: str = Query(...),
    limit: int = Query(10)
):
    dataset = clean_dataset_name(dataset)
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
    table_name = f"{dataset}_documents"

    if not os.path.exists(db_path):
        return {"error": "❌ قاعدة البيانات غير موجودة"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT doc_id, doc_text, clean_text FROM {table_name}
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return {
        "dataset": dataset,
        "documents": [
            {"doc_id": row[0], "doc_text": row[1], "clean_text": row[2]}
            for row in rows
        ]
    }

# ✅ عرض جميع الجداول داخل قاعدة البيانات
@app.get("/list-db-tables")
def list_db_tables():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

    if not os.path.exists(db_path):
        return {"error": "❌ قاعدة البيانات غير موجودة"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    return {
        "tables": [t[0] for t in tables]
    }

@app.get("/show-all-tables-content")
def show_all_tables_content(limit: int = 5):
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))

    if not os.path.exists(db_path):
        return {"error": "❌ قاعدة البيانات غير موجودة"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    result = {}

    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f"SELECT doc_id, doc_text, clean_text FROM {table_name} LIMIT ?", (limit,))
            rows = cursor.fetchall()
            result[table_name] = [
                {"doc_id": row[0], "doc_text": row[1], "clean_text": row[2]}
                for row in rows
            ]
        except Exception as e:
            result[table_name] = f"⚠️ خطأ أثناء القراءة: {str(e)}"

    conn.close()
    return result

@app.get("/show-table")
def show_table(
    dataset: str = Query(..., description="اسم مجموعة البيانات (مثلاً: trec_tot أو antique)"),
    limit: int = Query(10, description="عدد الوثائق التي سيتم عرضها")
):
    # تنظيف اسم المجموعة
    dataset = dataset.strip()
    # بناء اسم الجدول بناءً على اسم المجموعة
    table_name = f"{dataset}_documents"
    
    # بناء مسار قاعدة البيانات
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ir_docs.db'))
    
    if not os.path.exists(db_path):
        return {"error": "❌ قاعدة البيانات غير موجودة"}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT doc_id, doc_text, clean_text FROM {table_name} LIMIT ?", (limit,))
        rows = cursor.fetchall()
    except Exception as e:
        conn.close()
        return {"error": f"❌ حدث خطأ أثناء قراءة الجدول {table_name}: {str(e)}"}
    
    conn.close()
    
    return {
        "table": table_name,
        "documents": [
            {"doc_id": row[0], "doc_text": row[1], "clean_text": row[2]}
            for row in rows
        ]
    }
