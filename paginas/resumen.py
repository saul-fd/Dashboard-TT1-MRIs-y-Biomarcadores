import pandas as pd
import streamlit as st

import datos as d
import graficas as g

st.title("Clasificación de estados clínicos del Alzheimer")
st.markdown(
    "Dos modelos de aprendizaje automático clasifican a los participantes de ADNI en **CN** (cognitivamente normal), "
    "**MCI** (deterioro cognitivo leve) y **AD** (enfermedad de Alzheimer): uno con **biomarcadores en plasma** y datos "
    "demográficos, y otro con **resonancia magnética estructural (MRI)**. El énfasis está en detectar el MCI, la etapa temprana."
)

coh = d.cohorte()
preds = d.predicciones()
tabla = d.tabla_modelos()
mri = tabla.set_index("modelo").loc[d.MODELO_MRI]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sujetos en la cohorte", f"{len(coh):,}")
c2.metric("Conjunto de prueba", f"{(preds[d.MODELO_MRI]['particion'] == 'test').sum()} sujetos")
c3.metric("Exactitud balanceada (MRI)", f"{mri['test']:.1%}", help="DenseNet-121 3D en el conjunto de prueba")
c4.metric("Recall de MCI (MRI)", f"{mri['recall_MCI']:.1%}")

st.divider()
izq, der = st.columns([3, 2], gap="large")

with izq:
    st.subheader("Desempeño en el conjunto de prueba")
    st.caption("Exactitud balanceada con intervalo de confianza al 95% (bootstrap). La línea punteada es el azar en tres clases.")
    principales = tabla[tabla["modelo"].isin([d.MODELO_MRI, d.MODELO_PLASMA, "Fusión tardía 0.5·plasma + 0.5·MRI"])]
    st.plotly_chart(g.barras_ic(principales, "test", "modelo", "ic_inf", "ic_sup", destacar=principales["modelo"] == d.MODELO_MRI),
                    width="stretch")

with der:
    st.subheader("Recall por clase (MRI)")
    st.caption("Proporción de sujetos de cada clase que el modelo identifica correctamente.")
    rec = pd.DataFrame({"Clase": d.CLASES, "Recall": [mri[f"recall_{c}"] for c in d.CLASES], "Modelo": d.MODELO_MRI})
    st.plotly_chart(g.recall_por_clase(rec, "Clase", alto=330).update_layout(showlegend=False), width="stretch")

st.subheader("Hallazgos principales")
h1, h2, h3 = st.columns(3)
for col, titulo, texto in [
    (h1, "Hallazgo sobre el modelo de MRI", "Espacio para resumir el desempeño de la DenseNet-121 frente a la referencia."),
    (h2, "Hallazgo sobre la detección de MCI", "Espacio para describir qué tan bien se detecta la etapa temprana y con qué punto de operación."),
    (h3, "Hallazgo sobre la fusión", "Espacio para comparar plasma, MRI y su combinación."),
]:
    with col, st.container(border=True):
        st.markdown(f"**{titulo}.** {texto}")

st.info("Mockup con datos ficticios generados al azar: ninguna cifra corresponde a resultados reales.", icon=":material/info:")
