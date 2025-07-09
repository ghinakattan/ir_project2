import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from umap import UMAP
from scipy.sparse import load_npz

# === الإعدادات الأساسية ===
dataset = "trec_tot"  # أو "antique"
project_root = os.getcwd()

# حسب المسارات التي أرسلتها، ملفات tfidf ضمن offline_indexing_service
base_offline_indexing_path = os.path.join(project_root, "offline_indexing_service", "data", dataset)

vectorizer_path = os.path.join(base_offline_indexing_path, "tfidf_vectorizer.pkl")
tfidf_matrix_path = os.path.join(base_offline_indexing_path, "tfidf_docs_matrix.npz")

# ملفات الوثائق الأصلية والمنظفة ضمن مجلد data مباشرة (وليس ضمن offline_indexing_service)
base_data_path = os.path.join(project_root, "data", dataset)

if dataset == "trec_tot":
    doc_file = os.path.join(base_data_path, "trec_tot_docs_clean.csv")
elif dataset == "antique":
    doc_file = os.path.join(base_data_path, "antique_docs_clean.csv")
else:
    doc_file = None  # أو رفع استثناء هنا

print("vectorizer_path:", vectorizer_path)
print("tfidf_matrix_path:", tfidf_matrix_path)
print("doc_file:", doc_file)

# === تحميل البيانات ===
print("📦 تحميل النموذج والمصفوفة...")
vectorizer = joblib.load(vectorizer_path)
tfidf_matrix = load_npz(tfidf_matrix_path)

print(f"✅ شكل المصفوفة: {tfidf_matrix.shape}")
doc_ids = pd.read_csv(doc_file)["doc_id"].astype(str).tolist()

 # === تقليل الأبعاد باستخدام UMAP (للعرض فقط) ===
print("📉 تقليل الأبعاد باستخدام UMAP...")
sample_size = min(5000, tfidf_matrix.shape[0])
sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
sample_matrix = tfidf_matrix[sample_indices]

umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
reduced_data = umap.fit_transform(sample_matrix)

# === عنقدة باستخدام KMeans على كامل البيانات (وليس فقط العينة) ===
print("🔄 تنفيذ MiniBatchKMeans على كامل الوثائق...")
n_clusters = 5
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512, random_state=42)
full_labels = kmeans.fit_predict(tfidf_matrix)

# === عرض أهم الكلمات لكل عنقود (باستخدام كامل البيانات) ===
print("\n📌 أهم الكلمات لكل عنقود:")
terms = vectorizer.get_feature_names_out()
for i in range(n_clusters):
    cluster_indices = np.where(full_labels == i)[0]
    cluster_matrix = tfidf_matrix[cluster_indices].mean(axis=0).A1
    top_indices = cluster_matrix.argsort()[::-1][:10]
    top_terms = [terms[j] for j in top_indices]
    print(f"\nCluster {i}:")
    print(", ".join(top_terms))

# === حفظ نتائج الكلاستر لكل الوثائق ===
print("\n💾 حفظ نتائج التجميع...")
docs_df = pd.DataFrame({
    "doc_id": doc_ids,  # جميع الوثائق
    "cluster_label": full_labels
})
results_path = os.path.join(base_offline_indexing_path, "tfidf_clusters.csv")
docs_df.to_csv(results_path, index=False)
print(f"📁 تم حفظ نتائج التجميع في: {results_path}")

# === حفظ المخطط باستخدام العينة فقط (كما هو) ===
print("📊 رسم العنقود...")
plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap('tab10', n_clusters)
sample_labels = full_labels[sample_indices]  # خذ التصنيفات من كامل الوثائق

for i in range(n_clusters):
    cluster_points = reduced_data[sample_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"Cluster {i}", s=30, alpha=0.6, color=colors(i))

plt.title("🔵 توزيع الوثائق حسب العنقود")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(base_offline_indexing_path, "clusters_plot.png")
plt.savefig(plot_path)
plt.show()

# === حفظ نتائج التجميع كـ CSV ===
print("\n💾 حفظ نتائج التجميع...")
docs_df = pd.DataFrame({
    "doc_id": doc_ids,                 # جميع الوثائق
    "cluster_label": full_labels      # التصنيفات الجديدة لكل وثيقة
})
results_path = os.path.join(base_offline_indexing_path, "tfidf_clusters.csv")
docs_df.to_csv(results_path, index=False)
print(f"📁 تم حفظ نتائج التجميع في: {results_path}")

# === حفظ نموذج UMAP ===
umap_model_path = os.path.join(base_offline_indexing_path, "umap_model.joblib")
joblib.dump(umap, umap_model_path)
print(f"💾 تم حفظ نموذج UMAP في: {umap_model_path}")

# === حفظ نموذج الكلاستر KMeans ===
cluster_model_path = os.path.join(base_offline_indexing_path, "cluster_model.joblib")
joblib.dump(kmeans, cluster_model_path)
print(f"💾 تم حفظ نموذج KMeans في: {cluster_model_path}")

print(f"\n✅ التحليل اكتمل. تم حفظ المخطط في '{plot_path}'")
