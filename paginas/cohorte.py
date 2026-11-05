import plotly.express as px
import streamlit as st

import datos as d
import graficas as g

st.title("Cohorte de imágenes")
st.caption("Sujetos con resonancia T1 y, cuando existe, muestra de plasma cercana. Un renglón por sujeto.")

coh = d.cohorte()

f1, f2, f3, f4 = st.columns([2, 1.3, 1.3, 2])
clases = f1.multiselect("Diagnóstico", d.CLASES, default=d.CLASES)
sexo = f2.multiselect("Sexo", ["Mujer", "Hombre"], default=["Mujer", "Hombre"])
conjunto = f3.multiselect("Conjunto", sorted(coh["Conjunto"].unique()), default=sorted(coh["Conjunto"].unique()))
edad = f4.slider("Edad (años)", 55, 90, (55, 90))

df = coh[coh["Diagnóstico"].isin(clases) & coh["Sexo"].isin(sexo) & coh["Conjunto"].isin(conjunto) & coh["Edad"].between(*edad)]
if df.empty:
    st.warning("Ningún sujeto cumple con los filtros.")
    st.stop()

m = st.columns(4)
m[0].metric("Sujetos", f"{len(df):,}")
m[1].metric("Edad media", f"{df['Edad'].mean():.1f} años")
m[2].metric("Mujeres", f"{(df['Sexo'] == 'Mujer').mean():.0%}")
m[3].metric("IMC medio", f"{df['IMC'].mean():.1f}")

orden = {"Diagnóstico": d.CLASES, "Grupo de edad": d.ETIQUETAS_EDAD, "Rango de IMC": d.ETIQUETAS_IMC}


def proporciones(columna):
    tab = df.groupby([columna, "Diagnóstico"], observed=True).size().rename("Sujetos").reset_index()
    tab["Proporción"] = tab["Sujetos"] / tab.groupby(columna, observed=True)["Sujetos"].transform("sum")
    fig = px.bar(tab, x=columna, y="Proporción", color="Diagnóstico", color_discrete_map=g.COLOR_CLASE,
                 category_orders=orden, hover_data={"Sujetos": True, "Proporción": ":.0%"})
    fig.update_yaxes(tickformat=".0%")
    return g.estilo(fig)


a, b = st.columns(2, gap="large")
with a:
    st.subheader("Distribución de edad por diagnóstico")
    fig = px.box(df, x="Diagnóstico", y="Edad", color="Diagnóstico", color_discrete_map=g.COLOR_CLASE, category_orders=orden, points=False)
    st.plotly_chart(g.estilo(fig, leyenda=False), width="stretch")
with b:
    st.subheader("Diagnóstico por grupo de edad")
    st.plotly_chart(proporciones("Grupo de edad"), width="stretch")

a, b = st.columns(2, gap="large")
with a:
    st.subheader("Diagnóstico por sexo")
    tab = df.groupby(["Sexo", "Diagnóstico"], observed=True).size().rename("Sujetos").reset_index()
    fig = px.bar(tab, x="Sujetos", y="Sexo", color="Diagnóstico", orientation="h", color_discrete_map=g.COLOR_CLASE, category_orders=orden)
    st.plotly_chart(g.estilo(fig, alto=380), width="stretch")
with b:
    st.subheader("Diagnóstico por peso corporal (IMC)")
    st.plotly_chart(proporciones("Rango de IMC"), width="stretch")

st.subheader("Partición de los datos")
tab = df.groupby(["Partición", "Diagnóstico"], observed=True).size().rename("Sujetos").reset_index()
fig = px.bar(tab, x="Partición", y="Sujetos", color="Diagnóstico", barmode="group", color_discrete_map=g.COLOR_CLASE,
             category_orders={**orden, "Partición": ["Solo entrenamiento", "Entrenamiento/validación", "Prueba"]}, text_auto=True)
fig.update_traces(marker_cornerradius=4)
st.plotly_chart(g.estilo(fig, alto=320), width="stretch")
st.caption("La partición es por sujeto y la comparte el modelo de biomarcadores; los sujetos solo-MRI solo entran al entrenamiento.")

with st.expander("Ver tabla resumen por diagnóstico"):
    resumen = df.groupby("Diagnóstico", observed=True).agg(
        Sujetos=("Edad", "size"), Edad_media=("Edad", "mean"), IMC_medio=("IMC", "mean"),
        Mujeres=("Sexo", lambda s: (s == "Mujer").mean()))
    st.dataframe(resumen.style.format({"Edad_media": "{:.1f}", "IMC_medio": "{:.1f}", "Mujeres": "{:.0%}"}), width="stretch")
