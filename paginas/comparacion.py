import streamlit as st

import datos as d
import graficas as g

st.title("Comparación y fusión de modelos")
st.caption("Resonancia (DenseNet-121), plasma y su combinación sobre los mismos sujetos de prueba. "
           "La fusión tardía promedia las probabilidades; el apilamiento entrena un segundo modelo sobre ellas.")

tabla = d.tabla_modelos()
elegidos = st.multiselect("Modelos", tabla["modelo"].tolist(), default=tabla["modelo"].tolist())
df = tabla[tabla["modelo"].isin(elegidos)].sort_values("test", ascending=False)
if df.empty:
    st.warning("Elija al menos un modelo.")
    st.stop()

st.subheader("Exactitud balanceada en prueba")
st.plotly_chart(g.barras_ic(df, "test", "modelo", "ic_inf", "ic_sup", destacar=df["familia"] == "Fusión"), width="stretch")
st.caption("Azul: modelos de fusión. Gris: un solo tipo de dato.")

st.subheader("Recall por clase")
st.caption("Proporción de sujetos de cada clase que detecta cada modelo, en el mismo orden que la gráfica anterior.")
st.plotly_chart(g.mapa_recall(df.set_index("modelo")[["recall_CN", "recall_MCI", "recall_AD"]].set_axis(d.CLASES, axis=1)), width="stretch")

with st.expander("Ver tabla completa"):
    st.dataframe(df.style.format({c: "{:.1%}" for c in ["test", "ic_inf", "ic_sup", "recall_CN", "recall_MCI", "recall_AD"]}),
                 width="stretch", hide_index=True)
