import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import datos as d
import graficas as g

st.title("Modelo de resonancia magnética (MRI)")
st.caption(f"{d.MODELO_MRI}: red convolucional sobre el volumen completo del cerebro, preprocesado y registrado al espacio MNI152.")

desempeno, operacion, cerebro = st.tabs(["Desempeño", "Punto de operación de MCI", "Qué mira la red"])
pred = d.predicciones()[d.MODELO_MRI]

# ---------------------------------------------------------------- desempeño
with desempeno:
    particion = st.segmented_control("Conjunto", ["test", "cv"], default="test",
                                     format_func={"test": "Prueba", "cv": "Validación cruzada (fuera de fold)"}.get)
    y, prob, p = d.y_prob(pred, particion or "test")
    ba, inf, sup = d.ba_con_ic(y, p)
    rep = d.reporte_por_clase(y, prob, p)

    m = st.columns(4)
    m[0].metric("Exactitud balanceada", f"{ba:.1%}", help=f"IC 95%: {inf:.1%}–{sup:.1%}")
    m[1].metric("Recall de MCI", f"{rep.loc['MCI', 'Recall']:.1%}")
    m[2].metric("PR-AUC de MCI", f"{rep.loc['MCI', 'PR-AUC']:.1%}")
    m[3].metric("Sujetos evaluados", f"{len(y)}")

    a, b = st.columns([2, 3], gap="large")
    with a:
        st.subheader("Matriz de confusión")
        st.plotly_chart(g.matriz_confusion(d.matriz_confusion(y, p)), width="stretch")
    with b:
        st.subheader("Reporte por clase")
        st.dataframe(rep.style.format({c: "{:.1%}" for c in ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]}), width="stretch")
        st.subheader("Curva de entrenamiento")
        fig = px.line(d.historial(), x="epoch", y="val_bal_acc", labels={"epoch": "Época", "val_bal_acc": "BA en validación interna"})
        fig.update_traces(line_color=g.AZUL, line_width=2)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(g.estilo(fig, alto=230, leyenda=False), width="stretch")
        st.caption("Promedio de los cinco folds.")

# ---------------------------------------------------------------- punto de operación
with operacion:
    st.markdown("La probabilidad de MCI se multiplica por un factor para detectar más casos de la etapa temprana. "
                "Mueva el factor para ver cómo cambia el recall de cada clase.")
    peso = st.slider("Factor de la probabilidad de MCI", 0.5, 3.0, d.PESO_MCI_FINAL, 0.01)
    filas = []
    for part, etiqueta in [("cv", "Validación cruzada"), ("test", "Prueba")]:
        y, prob, p = d.y_prob(pred, part, peso)
        for k, c in enumerate(d.CLASES):
            filas.append({"Conjunto": etiqueta, "Clase": c, "Recall": (p[y == k] == k).mean()})
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.plotly_chart(g.recall_por_clase(pd.DataFrame(filas), "Conjunto"), width="stretch")
    with b:
        pesos = np.round(np.exp(np.linspace(np.log(0.5), np.log(3), 80)), 3)
        y, prob, _ = d.y_prob(pred, "test")
        curva = pd.DataFrame([{"Factor": w, "Clase": c, "Recall": ((prob * [1, w, 1]).argmax(1)[y == k] == k).mean()}
                              for w in pesos for k, c in enumerate(d.CLASES)])
        fig = px.line(curva, x="Factor", y="Recall", color="Clase", color_discrete_map=g.COLOR_CLASE, log_x=True)
        fig.update_traces(line_width=2)
        fig.add_vline(x=peso, line_color=g.GRIS, line_dash="dot")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(g.estilo(fig, alto=360), width="stretch")
        st.caption("Recall en prueba al variar el factor. La línea punteada marca el valor del deslizador.")

# ---------------------------------------------------------------- cerebro promedio y oclusión
with cerebro:
    st.subheader("Cerebro promedio por diagnóstico")
    st.caption("Promedio de las resonancias preprocesadas de cada clase. En el mockup se muestra un cerebro sintético.")
    vols = d.cerebros_promedio()
    c1, c2 = st.columns([1, 3])
    plano = c1.radio("Plano", ["Axial", "Coronal", "Sagital"])
    eje = {"Sagital": 0, "Coronal": 1, "Axial": 2}[plano]
    n = vols["CN"].shape[eje]
    idx = c2.slider("Corte", 0, n - 1, n // 2)
    cols = st.columns(4)
    for col, c in zip(cols, d.CLASES):
        col.markdown(f"**{c}**")
        col.plotly_chart(g.imagen_cerebro(g.corte(vols[c], eje, idx), zmin=0, zmax=255), width="stretch", key=f"cerebro_{c}")
    cols[3].markdown("**AD − CN**")
    cols[3].plotly_chart(g.imagen_cerebro(g.corte(vols["AD"] - vols["CN"], eje, idx), escala="RdBu", zmin=-120, zmax=120),
                         width="stretch", key="cerebro_dif")
    st.caption("En AD − CN, los tonos rojos indican menor intensidad en AD (atrofia y ventrículos más grandes).")

    st.subheader("Mapa de oclusión")
    st.caption("Se tapa un cubo del volumen a la vez y se mide cuánto cae la probabilidad de la clase correcta: "
               "las zonas naranjas son las que más usa el modelo para decidir.")
    clase = st.segmented_control("Clase", ["AD", "MCI"], default="AD") or "AD"
    calor = d.mapa_oclusion(clase)
    pico = np.unravel_index(calor.argmax(), calor.shape)
    cols = st.columns(3)
    for col, (plano, eje) in zip(cols, [("Sagital", 0), ("Coronal", 1), ("Axial", 2)]):
        col.markdown(f"**{clase} · {plano}**")
        col.plotly_chart(g.imagen_oclusion(g.corte(vols[clase], eje, pico[eje]), g.corte(calor, eje, pico[eje])),
                         width="stretch", key=f"oclusion_{plano}")
