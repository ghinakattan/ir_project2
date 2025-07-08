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
dataset = "trec_tot"  # أو antique
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_path = os.path.join(project_root, 'data', dataset)
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")
tfidf_matrix_path = os.path.join(base_path, "tfidf_docs_matrix.npz")
doc_file = os.path.join(base_path, "tot_docs_clean.csv")  # ✅ تم تعديله حسب طلبك

# === تحميل البيانات ===
print("📦 تحميل النموذج والمصفوفة...")
vectorizer = joblib.load(vectorizer_path)
tfidf_matrix = load_npz(tfidf_matrix_path)

print(f"✅ شكل المصفوفة: {tfidf_matrix.shape}")
doc_ids = pd.read_csv(doc_file)["doc_id"].astype(str).tolist()

# === تقليل الأبعاد باستخدام UMAP ===
print("📉 تقليل الأبعاد باستخدام UMAP...")
umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
sample_size = min(5000, tfidf_matrix.shape[0])
sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
sample_matrix = tfidf_matrix[sample_indices]
reduced_data = umap.fit_transform(sample_matrix)

# === عنقدة باستخدام KMeans ===
print("🔄 تنفيذ MiniBatchKMeans...")
n_clusters = 5
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512, random_state=42)
labels = kmeans.fit_predict(reduced_data)


# # ✅ أضف بعده هذا السطر 👇
# joblib.dump(umap, os.path.join(base_path, "umap_model.joblib"))
# joblib.dump(kmeans, os.path.join(base_path, "clusters_model.joblib"))


# === عرض الكلمات المفتاحية لكل عنقود ===
print("\n📌 أهم الكلمات لكل عنقود:")
terms = vectorizer.get_feature_names_out()
for i in range(n_clusters):
    print(f"\nCluster {i}:")
    cluster_indices = np.where(labels == i)[0]
    cluster_matrix = sample_matrix[cluster_indices].mean(axis=0).A1
    top_indices = cluster_matrix.argsort()[::-1][:10]
    top_terms = [terms[j] for j in top_indices]
    print(", ".join(top_terms))

# === تقييم العنقدة ===
print("\n📈 تقييم العنقدة:")
print(f"Silhouette Score: {silhouette_score(reduced_data, labels):.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(reduced_data, labels):.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(reduced_data, labels):.3f}")

# === رسم العنقود ===
print("📊 رسم العنقود...")
plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap('tab10', n_clusters)
for i in range(n_clusters):
    cluster_points = reduced_data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"Cluster {i}", s=30, alpha=0.6, color=colors(i))

plt.title("🔵 توزيع الوثائق حسب العنقود")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(base_path, "clusters_plot.png")
plt.savefig(plot_path)
plt.show()

print(f"✅ التحليل اكتمل. تم حفظ المخطط في '{plot_path}'")
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics import (
#     silhouette_score,
#     davies_bouldin_score,
#     calinski_harabasz_score
# )
# from umap import UMAP
# from scipy.sparse import load_npz

# # === الإعدادات الأساسية ===
# dataset = "antique"  # تم تغييره لـ antique
# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset))
# vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")
# tfidf_matrix_path = os.path.join(base_path, "tfidf_docs_matrix.npz")
# doc_file = os.path.join(base_path, f"{dataset}_docs_clean.csv")

# # === تحميل البيانات ===
# print("📦 تحميل النموذج والمصفوفة...")
# vectorizer = joblib.load(vectorizer_path)
# tfidf_matrix = load_npz(tfidf_matrix_path)

# print(f"✅ شكل المصفوفة: {tfidf_matrix.shape}")
# doc_ids = pd.read_csv(doc_file)["doc_id"].astype(str).tolist()

# # === تقليل الأبعاد باستخدام UMAP ===
# print("📉 تقليل الأبعاد باستخدام UMAP...")
# umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
# sample_size = min(5000, tfidf_matrix.shape[0])
# sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
# sample_matrix = tfidf_matrix[sample_indices]
# reduced_data = umap.fit_transform(sample_matrix)

# # === عنقدة باستخدام KMeans ===
# print("🔄 تنفيذ MiniBatchKMeans...")
# n_clusters = 5
# kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512, random_state=42)
# labels = kmeans.fit_predict(reduced_data)

# # === عرض الكلمات المفتاحية لكل عنقود ===
# print("\n📌 أهم الكلمات لكل عنقود:")
# terms = vectorizer.get_feature_names_out()
# for i in range(n_clusters):
#     print(f"\nCluster {i}:")
#     cluster_indices = np.where(labels == i)[0]
#     cluster_matrix = sample_matrix[cluster_indices].mean(axis=0).A1
#     top_indices = cluster_matrix.argsort()[::-1][:10]
#     top_terms = [terms[j] for j in top_indices]
#     print(", ".join(top_terms))

# # === تقييم العنقدة ===
# print("\n📈 تقييم العنقدة:")
# print(f"Silhouette Score: {silhouette_score(reduced_data, labels):.3f}")
# print(f"Davies-Bouldin Index: {davies_bouldin_score(reduced_data, labels):.3f}")
# print(f"Calinski-Harabasz Index: {calinski_harabasz_score(reduced_data, labels):.3f}")

# # === رسم العنقود ===
# print("📊 رسم العنقود...")
# plt.figure(figsize=(10, 7))
# colors = plt.cm.get_cmap('tab10', n_clusters)
# for i in range(n_clusters):
#     cluster_points = reduced_data[labels == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
#                 label=f"Cluster {i}", s=30, alpha=0.6, color=colors(i))

# plt.title("🔵 توزيع الوثائق حسب العنقود - Antique")
# plt.xlabel("UMAP-1")
# plt.ylabel("UMAP-2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(base_path, "clusters_plot_antique.png"))
# plt.show()

# print("✅ التحليل اكتمل. تم حفظ المخطط في 'clusters_plot_antique.png'")
