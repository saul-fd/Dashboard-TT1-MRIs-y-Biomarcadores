import plotly.express as px
import streamlit as st

import datos as d
import graficas as g

st.title("Modelo de biomarcadores")
st.caption("XGBoost con biomarcadores en plasma y datos demográficos, entrenado con la partición por sujeto compartida con el modelo de MRI.")

particion = st.segmented_control("Conjunto", ["test", "cv"], default="test",
                                 format_func={"test": "Prueba", "cv": "Validación cruzada (fuera de fold)"}.get)
y, prob, p = d.y_prob(d.predicciones()[d.MODELO_PLASMA], particion or "test")
ba, inf, sup = d.ba_con_ic(y, p)
rep = d.reporte_por_clase(y, prob, p)

m = st.columns(4)
m[0].metric("Exactitud balanceada", f"{ba:.1%}", help=f"IC 95%: {inf:.1%}–{sup:.1%}")
m[1].metric("Recall de CN", f"{rep.loc['CN', 'Recall']:.1%}")
m[2].metric("Recall de MCI", f"{rep.loc['MCI', 'Recall']:.1%}")
m[3].metric("Recall de AD", f"{rep.loc['AD', 'Recall']:.1%}")

a, b = st.columns([2, 3], gap="large")
with a:
    st.subheader("Matriz de confusión")
    st.plotly_chart(g.matriz_confusion(d.matriz_confusion(y, p)), width="stretch")
with b:
    st.subheader("Reporte por clase")
    st.dataframe(rep.style.format({c: "{:.1%}" for c in ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]}), width="stretch")
    st.caption("PR-AUC: área bajo la curva precisión-recall de cada clase contra el resto.")

st.subheader("Importancia de las variables")
st.caption("Contribución de cada variable a las decisiones del modelo.")
imp = d.importancia_variables().sort_values("Importancia")
fig = px.bar(imp, x="Importancia", y="Variable", orientation="h", text_auto=".1%")
fig.update_traces(marker_color=g.AZUL, marker_cornerradius=4)
fig.update_xaxes(tickformat=".0%")
st.plotly_chart(g.estilo(fig, alto=380, leyenda=False), width="stretch")
