import streamlit as st

st.title("Clasificación de estados clínicos del Alzheimer")
st.caption("Mockup inicial del dashboard · Trabajo Terminal · IPN-ESCOM")
st.info("Todas las cifras son ejemplos para ilustrar el diseño; no representan datos ni resultados reales.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sujetos en la cohorte", "1,200")
c2.metric("Conjunto de prueba", "240 sujetos")
c3.metric("Exactitud balanceada (MRI)", "82 %")
c4.metric("Recall de MCI (MRI)", "78 %")
