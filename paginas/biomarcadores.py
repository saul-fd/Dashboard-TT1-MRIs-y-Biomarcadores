import plotly.express as px
import streamlit as st

import datos as d
import graficas as g

st.title("Biomarcadores en plasma")
st.caption("Muestras de plasma con el diagnóstico más cercano en el tiempo.")

bio = d.biomarcadores()

f1, f2, f3, f4 = st.columns([2, 1.3, 2, 2])
marcador = f1.selectbox("Biomarcador", list(d.BIOMARCADORES), format_func=d.BIOMARCADORES.get)
sexo = f2.multiselect("Sexo", ["Mujer", "Hombre"], default=["Mujer", "Hombre"])
grupos = f3.multiselect("Grupo de edad", d.ETIQUETAS_EDAD, default=d.ETIQUETAS_EDAD)
lotes = f4.multiselect("Lote de medición", sorted(bio["Lote"].unique()), default=sorted(bio["Lote"].unique()))

df = bio[bio["Sexo"].isin(sexo) & bio["Grupo de edad"].isin(grupos) & bio["Lote"].isin(lotes)]
if df.empty:
    st.warning("Ninguna muestra cumple con los filtros.")
    st.stop()
nombre = d.BIOMARCADORES[marcador]
log = marcador != "AB42_AB40"

a, b = st.columns([3, 2], gap="large")
with a:
    st.subheader(f"{nombre} por diagnóstico")
    fig = px.box(df, x="Diagnóstico", y=marcador, color="Diagnóstico", color_discrete_map=g.COLOR_CLASE,
                 category_orders={"Diagnóstico": d.CLASES}, points=False, log_y=log, labels={marcador: nombre})
    if log:
        fig.update_yaxes(dtick="D2")
    st.plotly_chart(g.estilo(fig, alto=420, leyenda=False), width="stretch")
    if log:
        st.caption("Eje vertical en escala logarítmica.")
with b:
    st.subheader(f"Mediana de {nombre} por edad")
    tab = df.groupby(["Grupo de edad", "Diagnóstico"], observed=True)[marcador].median().rename(nombre).reset_index()
    fig = px.line(tab, x="Grupo de edad", y=nombre, color="Diagnóstico", markers=True, color_discrete_map=g.COLOR_CLASE,
                  category_orders={"Diagnóstico": d.CLASES, "Grupo de edad": d.ETIQUETAS_EDAD})
    fig.update_traces(line_width=2, marker_size=8)
    st.plotly_chart(g.estilo(fig, alto=420), width="stretch")

st.subheader("¿Qué biomarcador separa mejor cada diagnóstico?")
st.caption("AUC univariada de cada biomarcador para distinguir una clase del resto (0.5 = no separa, 1 = separa perfectamente). "
           "Se calcula con todas las muestras, sin filtros.")
fig = px.bar(d.poder_discriminante(), x="Biomarcador", y="AUC", color="Clase", barmode="group", color_discrete_map=g.COLOR_CLASE,
             category_orders={"Clase": d.CLASES}, text_auto=".2f")
fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False)
fig.update_yaxes(range=[0.5, 1])
fig.update_xaxes(title=None)
st.plotly_chart(g.estilo(fig, alto=400), width="stretch")
