#langkah 3 - aplikasi streamlit (membuat aplikasi analisis sentimen)
import joblib
import streamlit as st
from preprocess import bersihkan

st.set_page_config(page_title="Analisis Sentimen Ulasan", page_icon="R")
st.title("Analisis Sentimen Ulasan Aplikasi Gojek")
st.write("Ketik ulasan, model akan memprediksi positif atau negatif.")

@st.cache_resource
def muat_model():
    return joblib.load("model.joblib")

model = muat_model()
teks = st.text_area("Masukkan ulasan:", placeholder="Contoh: aplikasinya bagus dan cepat")

if st.button("Prediksi"):
    if not teks.strip():
        st.warning("Isi ulasan dulu.")
    else:
        bersih = bersihkan(teks)
        proba = model.predict_proba([bersih])[0]
        kelas = model.classes_
        idx = proba.argmax()
        if kelas[idx] == "positif":
            st.success(f"Positif ({proba[idx]:.1%})")
        else:
            st.error(f"Negatif ({proba[idx]:.1%})")
        st.bar_chart(dict(zip(kelas, proba)))
        with st.expander("Teks setelah preprocessing"):
            st.code(bersih)
