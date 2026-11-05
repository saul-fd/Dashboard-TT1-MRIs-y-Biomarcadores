"""Módulo "Crear dashboard": punto de entrada del dashboard del Trabajo Terminal (mockup con datos ficticios).

Ejecutar desde la raíz del proyecto:
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(page_title="TT Alzheimer · CN / MCI / AD", page_icon=":material/neurology:", layout="wide")

paginas = {
    "Proyecto": [
        st.Page("paginas/resumen.py", title="Resumen", icon=":material/dashboard:", default=True),
    ],
    "Exploración de datos": [
        st.Page("paginas/demograficos.py", title="Datos demográficos", icon=":material/badge:"),
        st.Page("paginas/cohorte.py", title="Cohorte de imágenes", icon=":material/groups:"),
        st.Page("paginas/biomarcadores.py", title="Biomarcadores en plasma", icon=":material/bloodtype:"),
        st.Page("paginas/evolucion.py", title="Evolución de la enfermedad", icon=":material/timeline:"),
    ],
    "Modelos": [
        st.Page("paginas/modelo_plasma.py", title="Modelo de biomarcadores", icon=":material/science:"),
        st.Page("paginas/modelo_mri.py", title="Modelo de resonancia (MRI)", icon=":material/neurology:"),
        st.Page("paginas/comparacion.py", title="Comparación y fusión", icon=":material/compare_arrows:"),
    ],
    "Información": [
        st.Page("paginas/acerca.py", title="Acerca del proyecto", icon=":material/info:"),
    ],
}

with st.sidebar:
    st.markdown("### Detección temprana de Alzheimer")
    st.caption("Trabajo Terminal · Lic. en Ciencia de Datos · IPN-ESCOM")
    st.warning("Mockup: todas las cifras y gráficas usan datos ficticios.", icon=":material/construction:")

st.navigation(paginas).run()
