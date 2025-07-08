from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import numpy as np
import os
from ranx import Qrels, Run, evaluate
import json

app = FastAPI()

@app.get("/evaluate")
async def evaluate_run(
    dataset: str = Query(..., description="اسم مجموعة البيانات (مثلاً: trec_tot)"),
    qrels_file: str = Query(..., description="اسم ملف qrels (مثل: trec_tot_qrels.csv)"),
    run_file: str = Query(..., description="اسم ملف run (مثل: tfidf_run_top10.csv)"),
    top_k: int = Query(10, description="عدد النتائج الأعلى لتقييمها"),
    results_file: str = Query(None, description="(اختياري) اسم ملف JSON لحفظ النتائج")
):
    try:
        # 🟡 المسار الحالي لملف app.py (evaluation_service)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # ✅ مسار ملف qrels: ضمن مجلد data/{dataset}
        qrels_path = os.path.normpath(os.path.join(current_dir, "..", "data", dataset, qrels_file))

        # ✅ مسار ملف run: ضمن مجلد offline_indexing_service/data/{dataset}
        run_path = os.path.normpath(os.path.join(current_dir, "..", "offline_indexing_service", "data", dataset, run_file))

        if not os.path.exists(qrels_path):
            raise HTTPException(status_code=404, detail=f"❌ Qrels file not found: {qrels_path}")
        if not os.path.exists(run_path):
            raise HTTPException(status_code=404, detail=f"❌ Run file not found: {run_path}")

        # قراءة الملفات
        df_qrels = pd.read_csv(qrels_path)
        df_run = pd.read_csv(run_path)

        # معالجة الأعمدة
        df_qrels["query_id"] = df_qrels["query_id"].astype(str)
        df_qrels["doc_id"] = df_qrels["doc_id"].astype(str)
        df_qrels["relevance"] = df_qrels["relevance"].apply(lambda x: int(float(x)))

        df_run["query_id"] = df_run["query_id"].astype(str)
        df_run["doc_id"] = df_run["doc_id"].astype(str)
        df_run["score"] = df_run["score"].astype(float)

        # تحويل إلى كائنات ranx
        qrels = Qrels.from_df(df_qrels, q_id_col="query_id", doc_id_col="doc_id", score_col="relevance")
        run = Run.from_df(df_run, q_id_col="query_id", doc_id_col="doc_id", score_col="score")

        # التقييم
        metrics = ["map", "mrr", f"precision@{top_k}", "recall"]
        results = evaluate(qrels, run, metrics=metrics, make_comparable=True)

        # ✅ حفظ النتائج داخل نفس مجلد offline_indexing_service/data/{dataset}
        results_file = results_file or f"{dataset}_eval_results.json"
        results_path = os.path.join(os.path.dirname(run_path), results_file)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        return {
            "✅": "تم التقييم بنجاح",
            "dataset": dataset,
            "saved_to": results_path,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
