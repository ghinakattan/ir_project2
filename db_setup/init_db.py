import os
import sqlite3
import pandas as pd

# المسار إلى مجلد المشروع الأساسي
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
db_path = os.path.join(data_dir, "ir_docs.db")

# ✅ تأكد من أن مجلد data موجود
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# الاتصال بقاعدة البيانات وإنشاء الجدول
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# إنشاء الجدول
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT,
    doc_id TEXT,
    doc_text TEXT,
    clean_text TEXT
)
""")

# تحميل البيانات من كل مجموعة بيانات
datasets = {
    "trec_tot": "data/trec_tot/tot_docs_clean.csv",
    "antique": "data/antique/antique_docs_clean.csv"
}

for dataset, path in datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            cursor.execute("""
            INSERT INTO documents (dataset, doc_id, doc_text, clean_text)
            VALUES (?, ?, ?, ?)
            """, (
                dataset,
                str(row.get("doc_id", "")),
                str(row.get("doc_text", "")),
                str(row.get("clean_text", ""))
            ))

conn.commit()
conn.close()
print("✅ تم إنشاء القاعدة واستيراد الوثائق بنجاح.")
