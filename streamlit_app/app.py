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
