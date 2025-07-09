# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# from st_aggrid import AgGrid, GridOptionsBuilder

# # --- إعداد الصفحة ---
# st.set_page_config(page_title="IR System", page_icon="🔍", layout="wide")

# # --- تنسيق CSS مخصص ---
# st.markdown("""
#     <style>
# .main {
#     background-color: #e6ebe6; /* رمادي دافئ فاتح */
#     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
# }
# .title {
#     font-size: 36px;
#     text-align: center;
#     color: #276749; /* أخضر داكن */
#     font-weight: bold;
# }
# .subtitle {
#     font-size: 18px;
#     color: #444;
#     text-align: center;
#     margin-bottom: 30px;
# }
# .stButton>button {
#     background-color: #276749; /* أخضر داكن */
#     color: white;
#     font-weight: bold;
#     width: 100%;
#     border-radius: 6px;
#     padding: 10px;
# }
# .stTextInput>div>div>input {
#     font-size: 16px;
# }
# h2, h3, .stTextInput label {
#     color: #276749;
# }

# .stButton>button {
#     background-color: #3a9950;  /* أخضر متوسط */
#     color: white;
#     font-weight: bold;
#     width: 100%;
#     border-radius: 6px;
#     padding: 10px;
#     border: none;
#     transition: background-color 0.3s ease;
# }
# .stButton>button:hover {
#     background-color: #2a6f34;  /* أخضر أغمق عند التمرير */
# }
# /* تحسين لون خطوط الجدول */
# .ag-theme-balham {
#     --ag-border-color: #d1d5db;
#     --ag-row-hover-color: #e6f0ea;
# }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("<div class='title'>🔍 Information Retrieval System</div>", unsafe_allow_html=True)
# st.markdown("---")

# # --- الشريط الجانبي ---
# with st.sidebar:
#     st.header("⚙️ إعدادات البحث")
#     dataset = st.selectbox("📁 اختر مجموعة البيانات", ["trec_tot", "antique"])
#     method = st.selectbox("📌 نوع التمثيل", ["tfidf", "embedding", "hybrid"])
#     top_k = st.slider("🔢 عدد النتائج", min_value=1, max_value=20, value=5)
#     save_results = st.checkbox("💾 حفظ النتائج؟")

# # --- إدخال الاستعلام ---
# st.subheader("📝 أدخل استعلامك")
# query_text = st.text_input("", placeholder="مثال: what is artificial intelligence?")

# # --- تعيين عناوين الخدمات حسب البورت ---
# base_urls = {
#     "tfidf": "http://127.0.0.1:8001",
#     "embedding": "http://127.0.0.1:8002",
#     "hybrid": "http://127.0.0.1:8003",
# }

# endpoint_map = {
#     "tfidf": "/query-match",
#     "embedding": "/query-embedding",
#     "hybrid": "/query-hybrid"
# }

# # خدمة تحسين الاستعلام على بورت منفصل
# refine_base_url = "http://127.0.0.1:8000"

# # --- تنفيذ البحث ---
# if st.button("🚀 تنفيذ البحث"):
#     if not query_text.strip():
#         st.warning("⚠️ الرجاء إدخال استعلام أولاً.")
#     else:
#         with st.spinner("🔄 جاري إرسال الاستعلام إلى الخدمة..."):
#             base_url = base_urls[method]
#             endpoint = endpoint_map[method]
#             payload = {
#                 "query": query_text,
#                 "top_k": top_k
#             }

#             try:
#                 response = requests.post(f"{base_url}{endpoint}?dataset={dataset}", json=payload)

#                 if response.status_code == 200:
#                     data = response.json()
#                     st.success("✅ تم تنفيذ الاستعلام بنجاح")
#                     results_df = pd.DataFrame(data["results"])

#                     # ✅ عرض تفاعلي للنتائج
#                     st.subheader("📋 النتائج:")
#                     gb = GridOptionsBuilder.from_dataframe(results_df)
#                     gb.configure_pagination()
#                     gb.configure_default_column(editable=False, groupable=True)
#                     AgGrid(results_df, gridOptions=gb.build(), theme="balham", height=350)

#                     # 📊 رسم بياني
#                     if "score" in results_df.columns:
#                         st.subheader("📈 توزيع درجات التشابه بين الوثائق والاستعلام")
#                         fig = px.bar(results_df, x="doc_id", y="score", color="score",
#                                      labels={"doc_id": "معرّف الوثيقة", "score": "درجة التشابه"},
#                                      color_continuous_scale=px.colors.sequential.Tealgrn)
#                         st.plotly_chart(fig, use_container_width=True)

#                         st.markdown(
#                             """
#                             <div style='font-size:14px; color:#555; margin-top:-10px; margin-bottom:30px;'>
#                             هذا الرسم البياني يوضح مدى تشابه كل وثيقة مع الاستعلام المدخل.  
#                             كل عمود يمثل وثيقة وارتفاعه يدل على درجة التشابه (كلما زادت الدرجة، كانت الوثيقة أكثر ملائمة للاستعلام).
#                             </div>
#                             """, unsafe_allow_html=True
#                         )

#                     # 💾 حفظ النتائج
#                     if save_results:
#                         filename = f"{dataset}_{method}_query_results.csv"
#                         results_df.to_csv(filename, index=False, encoding="utf-8")
#                         st.success(f"💾 تم حفظ النتائج في: `{filename}`")

#                 else:
#                     st.error(f"❌ حدث خطأ في الاستجابة:\n\n{response.text}")

#             except Exception as e:
#                 st.error(f"🚫 خطأ أثناء الاتصال بالخادم:\n\n{e}")

# # --- تحسين الاستعلام ---
# if st.button("🛠 تحسين الاستعلام"):
#     if not query_text.strip():
#         st.warning("⚠️ الرجاء إدخال استعلام أولاً.")
#     else:
#         with st.spinner("🔄 جاري إرسال الاستعلام للتحسين..."):
#             payload = {
#                 "query": query_text,
#                 "options": {
#                     "spelling_correction": True,
#                     "query_expansion": True,
#                     "query_suggestion": True
#                 }
#             }
#             try:
#                 response = requests.post(f"{refine_base_url}/refine-query?dataset={dataset}", json=payload)
#                 if response.status_code == 200:
#                     data = response.json()
#                     st.success("✅ تم تحسين الاستعلام بنجاح")

#                     st.markdown(f"**الاستعلام الأصلي:** `{data['original_query']}`")

#                     if data.get("cleaned_query") and data["cleaned_query"] != data['original_query']:
#                         st.markdown(f"**الاستعلام بعد التنظيف الإملائي:** `{data['cleaned_query']}`")

#                     if data.get("corrected_query") and data["corrected_query"] != data['original_query']:
#                         st.markdown(f"**الاستعلام بعد التصحيح الإملائي:** `{data['corrected_query']}`")
                        
#                     if data.get("expanded_query") and data["expanded_query"] != (data.get("corrected_query") or data['original_query']):
#                         st.markdown(f"**الاستعلام بعد التوسيع:** `{data['expanded_query']}`")

#                     if data.get("expanded_terms_added") and data["expanded_terms_added"] != (data.get("corrected_query") or data['original_query']):
#                         st.markdown(f"**الاستعلام بعد التبديل:** `{data['expanded_terms_added']}`")

                        
#                     if data.get("suggestions"):
#                         st.markdown("**اقتراحات لاستعلامات أخرى:**")
#                         for s in data["suggestions"]:
#                             st.markdown(f"- {s}")

#                 else:
#                     st.error(f"❌ حدث خطأ في الاستجابة:\n\n{response.text}")

#             except Exception as e:
#                 st.error(f"🚫 خطأ أثناء الاتصال بالخادم:\n\n{e}")
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="محرك البحث - TF-IDF - Embedding - Hybrid", layout="wide")

API_BASE_URL = "http://localhost:8000"  # عدل حسب عنوان سيرفر FastAPI لديك

def query_search(query: str, dataset: str, top_k: int, mode: str):
    """
    يستدعي API البحث حسب وضع البحث (tfidf, embedding, hybrid)
    """
    endpoint_map = {
        "tfidf": "/query_match",
        "embedding": "/query-embedding",
        "hybrid": "/query-hybrid"
    }
    url = API_BASE_URL + endpoint_map.get(mode, "/query_match")
    payload = {"query": query, "dataset": dataset, "top_k": top_k}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"خطأ في الاتصال بالخادم: {e}")
        return None

def main():
    st.title("محرك بحث النصوص")
    st.write("")

    # مدخلات المستخدم
    query_text = st.text_area("ادخل استعلام البحث", height=80)

    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox(
            "اختر مجموعة البيانات",
            options=["trec_tot", "antique"],
            index=0
        )
    with col2:
        top_k = st.number_input("عدد النتائج المراد عرضها", min_value=1, max_value=50, value=10, step=1)

    mode = st.radio(
        "نوع التمثيل:",
        options=["tfidf", "embedding", "hybrid"],
        index=0,
        horizontal=True
    )

    colb1, colb2 = st.columns(2)
    with colb1:
        search_btn = st.button("🚀 تنفيذ البحث")
    with colb2:
        improve_btn = st.button("🛠 تحسين الاستعلام")

    # مساحة عرض النتائج
    results_placeholder = st.empty()
    cleaned_query_placeholder = st.empty()

    if search_btn:
        if not query_text.strip():
            st.warning("⚠️ الرجاء إدخال استعلام أولاً.")
        else:
            with st.spinner("جاري تنفيذ البحث..."):
                data = query_search(query_text, dataset, top_k, mode)
                if data:
                    # عرض النص بعد التنظيف
                    cleaned_query = data.get("cleaned_query", "")
                    if cleaned_query:
                        cleaned_query_placeholder.markdown(f"**النص بعد التنظيف:** {cleaned_query}")

                    # عرض النتائج
                    results = data.get("results", [])
                    if results:
                        df = pd.DataFrame(results)
                        # عرض الجدول بشكل قابل للتمرير
                        results_placeholder.dataframe(df)
                    else:
                        results_placeholder.info("لا توجد نتائج للعرض.")
                else:
                    results_placeholder.error("فشل في الحصول على النتائج من الخادم.")

    elif improve_btn:
        if not query_text.strip():
            st.warning("⚠️ الرجاء إدخال استعلام أولاً لتحسينه.")
        else:
            with st.spinner("جاري تحسين الاستعلام..."):
                try:
                    refine_api_url = "http://localhost:8001/refine-query"  # رابط خدمة تحسين الاستعلام

                    payload = {
                        "query": query_text,
                        "options": {
                            "spelling_correction": True,
                            "query_expansion": True,
                            "query_suggestion": True
                        }
                    }
                    params = {"dataset": dataset}  # dataset كـ query param

                    response = requests.post(refine_api_url, params=params, json=payload)
                    response.raise_for_status()
                    improved = response.json()

                    refined_query = improved.get("refined_query", "")
                    if refined_query:
                        st.success("✅ تم تحسين الاستعلام بنجاح!")

                        with st.expander("📄 تفاصيل الاستعلام"):
                            st.markdown("**النص الأصلي:**")
                            st.write(improved.get("original_query", ""))

                            st.markdown("**النص بعد التنظيف:**")
                            st.write(improved.get("cleaned_query", ""))

                            if improved.get("corrected_query"):
                                st.markdown("**النص بعد التصحيح الإملائي:**")
                                st.write(improved.get("corrected_query"))

                        with st.expander("📝 الكلمات المصححة"):
                            corrected_terms = improved.get("corrected_terms_diff", [])
                            if corrected_terms:
                                for term in corrected_terms:
                                    st.markdown(f"- {term}")
                            else:
                                st.write("لا توجد كلمات مصححة.")

                        with st.expander("➕ الكلمات الموسعة"):
                            expanded_terms = improved.get("expanded_terms_added", [])
                            if expanded_terms:
                                for term in expanded_terms:
                                    st.markdown(f"- {term}")
                            else:
                                st.write("لا توجد كلمات موسعة.")

                        with st.expander("💡 اقتراحات استعلامات بديلة"):
                            suggestions = improved.get("suggestions", [])
                            if suggestions:
                                for sug in suggestions:
                                    st.markdown(f"- {sug}")
                            else:
                                st.write("لا توجد اقتراحات.")

                        st.markdown("**🔍 الاستعلام المحسن النهائي:**")
                        st.code(refined_query, language="text")
                    else:
                        st.info("لم يتم تحسين الاستعلام.")

                except Exception as e:
                    st.error(f"❌ خطأ في الاتصال بخدمة تحسين الاستعلام (8001): {e}")

if __name__ == "__main__":
    main()
