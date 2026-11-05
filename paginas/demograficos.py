import plotly.express as px
import streamlit as st

import datos as d
import graficas as g

st.title("Datos demográficos")
st.caption("Diagnósticos de ADNI (DXSUM) cruzados con los datos demográficos de los participantes (PTDEMOG), "
           "con la estructura de `Diagnosis_demograficos.csv` del análisis exploratorio del equipo de biomarcadores.")

dem = d.demograficos()

f1, f2, f3, f4 = st.columns([2.2, 1.8, 1.3, 2])
fases = f1.multiselect("Fase de ADNI", d.FASES, default=d.FASES)
clases = f2.multiselect("Diagnóstico", d.CLASES, default=d.CLASES)
sexo = f3.multiselect("Sexo", ["Mujer", "Hombre"], default=["Mujer", "Hombre"])
edad = f4.slider("Edad en la visita (años)", 55, 95, (55, 95))

df = dem[dem["Fase ADNI"].isin(fases) & dem["Diagnóstico"].isin(clases) & dem["Sexo"].isin(sexo) & dem["Edad"].between(*edad)]
if df.empty:
    st.warning("Ninguna visita cumple con los filtros.")
    st.stop()
# Las variables fijas por participante se cuentan una vez por persona (su última visita dentro del filtro)
part = df.sort_values("Visita").groupby("Participante").last()

m = st.columns(4)
m[0].metric("Participantes", f"{len(part):,}")
m[1].metric("Visitas con diagnóstico", f"{len(df):,}")
m[2].metric("Edad media", f"{part['Edad'].mean():.1f} años")
m[3].metric("Escolaridad media", f"{part['Escolaridad (años)'].mean():.1f} años")

orden = {"Diagnóstico": d.CLASES, "Fase ADNI": d.FASES}

a, b = st.columns(2, gap="large")
with a:
    st.subheader("Diagnóstico por fase de ADNI")
    tab = df.groupby(["Fase ADNI", "Diagnóstico"], observed=True).size().rename("Visitas").reset_index()
    fig = px.bar(tab, x="Fase ADNI", y="Visitas", color="Diagnóstico", color_discrete_map=g.COLOR_CLASE, category_orders=orden)
    st.plotly_chart(g.estilo(fig), width="stretch")
with b:
    st.subheader("Visitas por año de examen")
    tab = df.groupby(["Año del examen", "Diagnóstico"], observed=True).size().rename("Visitas").reset_index()
    fig = px.bar(tab, x="Año del examen", y="Visitas", color="Diagnóstico", color_discrete_map=g.COLOR_CLASE, category_orders=orden)
    st.plotly_chart(g.estilo(fig), width="stretch")

a, b = st.columns(2, gap="large")
with a:
    st.subheader("Escolaridad por diagnóstico")
    fig = px.box(part, x="Diagnóstico", y="Escolaridad (años)", color="Diagnóstico", color_discrete_map=g.COLOR_CLASE,
                 category_orders=orden, points=False)
    st.plotly_chart(g.estilo(fig, leyenda=False), width="stretch")
    st.caption("Diagnóstico de la última visita de cada participante.")
with b:
    st.subheader("Visitas por participante")
    visitas = df.groupby("Participante").size().value_counts().sort_index().rename_axis("Visitas").reset_index(name="Participantes")
    fig = px.bar(visitas, x="Visitas", y="Participantes", text_auto=True)
    fig.update_traces(marker_color=g.AZUL, marker_cornerradius=4)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(g.estilo(fig, leyenda=False), width="stretch")

st.subheader("Diagnóstico según otras variables demográficas")
c1, c2 = st.columns([2, 3])
variable = c1.selectbox("Variable", list(d.VARIABLES_DEMOGRAFICAS),
                        format_func=lambda v: f"{v} ({d.VARIABLES_DEMOGRAFICAS[v][1]})")
modo = c2.segmented_control("Mostrar", ["Proporción", "Participantes"], default="Proporción") or "Proporción"
categorias = list(d.VARIABLES_DEMOGRAFICAS[variable][0])
tab = part.groupby([variable, "Diagnóstico"], observed=True).size().rename("Participantes").reset_index()
tab["Proporción"] = tab["Participantes"] / tab.groupby(variable)["Participantes"].transform("sum")
fig = px.bar(tab, x=modo, y=variable, color="Diagnóstico", orientation="h", color_discrete_map=g.COLOR_CLASE,
             category_orders={**orden, variable: categorias[::-1]}, hover_data={"Participantes": True, "Proporción": ":.0%"})
if modo == "Proporción":
    fig.update_xaxes(tickformat=".0%")
st.plotly_chart(g.estilo(fig, alto=110 + 45 * len(categorias)), width="stretch")
st.caption("Entre paréntesis, el nombre de la variable en PTDEMOG.")

st.subheader("Tiempo desde el inicio de los síntomas cognitivos")
st.caption("Años entre el inicio estimado de los síntomas (PTCOGBEG) y cada visita; solo participantes con síntomas reportados.")
sint = df.dropna(subset=["Años desde inicio de síntomas"])
fig = px.box(sint, x="Diagnóstico", y="Años desde inicio de síntomas", color="Diagnóstico", color_discrete_map=g.COLOR_CLASE,
             category_orders=orden, points=False)
st.plotly_chart(g.estilo(fig, alto=340, leyenda=False), width="stretch")
