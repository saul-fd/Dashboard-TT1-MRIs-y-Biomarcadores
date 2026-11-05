import streamlit as st

st.title("Acerca del proyecto")

a, b = st.columns([3, 2], gap="large")
with a:
    st.subheader("Objetivo general")
    st.markdown("Implementar modelos de aprendizaje automático de clasificación de estados clínicos del Alzheimer en adultos "
                "de 55 a 90 años, para buscar una detección temprana, a través del análisis de imágenes de resonancia magnética, "
                "biomarcadores clínicos y datos demográficos de pacientes del estudio ADNI.")
    st.subheader("Cómo se construyó")
    st.markdown("""
1. **Datos.** Resonancias T1, biomarcadores en plasma, diagnósticos y demografía de ADNI. Cada muestra de plasma
   se empareja con la resonancia más cercana en el tiempo.
2. **Partición por sujeto**, compartida por ambos modelos, para que ningún paciente esté en entrenamiento y prueba a la vez.
3. **Modelo de biomarcadores:** XGBoost con variables de plasma y demográficas.
4. **Modelo de MRI:** preprocesamiento (corrección de intensidad, extracción de cráneo, registro a MNI152) y DenseNet-121 3D,
   con validación cruzada y evaluación en un conjunto de prueba aislado.
5. **Fusión y dashboard:** integración de resultados, estadísticas comparativas y gráficas de desempeño.
""")
with b:
    st.subheader("Glosario")
    st.markdown("""
- **CN:** cognitivamente normal.
- **MCI:** deterioro cognitivo leve, etapa temprana.
- **AD:** enfermedad de Alzheimer.
- **Exactitud balanceada:** promedio del recall de las tres clases; no se infla con la clase más frecuente.
- **Recall:** proporción de sujetos de una clase que el modelo detecta.
- **PR-AUC:** área bajo la curva precisión-recall de una clase contra el resto.
""")
    st.subheader("Privacidad")
    st.markdown("El dashboard no muestra identificadores de pacientes, identificadores de imagen ni fechas. Los casos "
                "longitudinales llevan un código anónimo y las imágenes son promedios por clase.")

st.divider()
st.caption("Datos obtenidos de la Alzheimer's Disease Neuroimaging Initiative (ADNI) (adni.loni.usc.edu). "
           "Esta versión es un mockup: todas las cifras y gráficas usan datos ficticios.")
