import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import datos as d
import graficas as g

st.title("Evolución de la enfermedad")
st.caption("Diagnóstico de cada sujeto en todas sus visitas de seguimiento. "
           "Los casos tienen un código anónimo y el tiempo se mide en años desde la primera visita.")

tr = d.transiciones()

f1, f2, f3 = st.columns([1.3, 2, 2])
sexo = f1.multiselect("Sexo", ["Mujer", "Hombre"], default=["Mujer", "Hombre"])
grupos = f2.multiselect("Edad en la primera visita", d.ETIQUETAS_EDAD, default=d.ETIQUETAS_EDAD)
seg = f3.slider("Seguimiento mínimo (años)", 0.0, 10.0, 1.0, 0.5)
df = tr[tr["Sexo"].isin(sexo) & tr["Grupo de edad"].isin(grupos) & (tr["Seguimiento (años)"] >= seg)]
if df.empty:
    st.warning("Ningún caso cumple con los filtros.")
    st.stop()

m = st.columns(4)
m[0].metric("Casos con seguimiento", f"{len(df):,}")
m[1].metric("Seguimiento mediano", f"{df['Seguimiento (años)'].median():.1f} años")
cn = df[df["Inicial"] == "CN"]
mci = df[df["Inicial"] == "MCI"]
m[2].metric("CN que progresan", f"{cn['Progresó'].mean():.0%}" if len(cn) else "—", help="A MCI o AD en su última visita")
m[3].metric("MCI que progresan a AD", f"{(mci['Final'] == 'AD').mean():.0%}" if len(mci) else "—")

a, b = st.columns([3, 2], gap="large")
with a:
    st.subheader("Diagnóstico inicial → diagnóstico final")
    flujo = df.groupby(["Inicial", "Final"], observed=True).size().reset_index(name="n")
    nodos = [f"{c} inicial" for c in d.CLASES] + [f"{c} final" for c in d.CLASES]
    color_nodo = [g.COLOR_CLASE[c] for c in d.CLASES] * 2
    fig = go.Figure(go.Sankey(
        node=dict(label=nodos, color=color_nodo, pad=18, thickness=16, line_width=0),
        link=dict(source=[d.CLASES.index(i) for i in flujo["Inicial"]], target=[3 + d.CLASES.index(f) for f in flujo["Final"]],
                  value=flujo["n"], color=[g.rgba(g.COLOR_CLASE[i], 0.35) for i in flujo["Inicial"]],
                  hovertemplate="%{source.label} → %{target.label}: %{value} casos<extra></extra>"),
    ))
    st.plotly_chart(g.estilo(fig, alto=400, leyenda=False), width="stretch")
with b:
    st.subheader("Tasa de progresión por edad")
    tab = (df[df["Inicial"] != "AD"].groupby(["Grupo de edad", "Inicial"], observed=True)["Progresó"]
           .agg(["mean", "size"]).reset_index().rename(columns={"mean": "Progresión", "size": "Casos", "Inicial": "Diagnóstico inicial"}))
    fig = px.bar(tab, x="Grupo de edad", y="Progresión", color="Diagnóstico inicial", barmode="group",
                 color_discrete_map=g.COLOR_CLASE, category_orders={"Grupo de edad": d.ETIQUETAS_EDAD, "Diagnóstico inicial": ["CN", "MCI"]},
                 hover_data={"Casos": True, "Progresión": ":.0%"})
    fig.update_traces(marker_cornerradius=4)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(g.estilo(fig, alto=400), width="stretch")

st.subheader("Análisis de casos")
st.caption("Elija un tipo de trayectoria y un caso para ver su diagnóstico en cada visita.")
c1, c2 = st.columns([1, 2])
tipos = df.assign(Trayectoria=df["Inicial"].astype(str) + " → " + df["Final"].astype(str))
conteo = tipos["Trayectoria"].value_counts()
tipo = c1.selectbox("Trayectoria", conteo.index, format_func=lambda t: f"{t}  ({conteo[t]} casos)")
casos = tipos.loc[tipos["Trayectoria"] == tipo].sort_values("Visitas", ascending=False)
caso = c2.selectbox("Caso", casos["Caso"], format_func=lambda c: f"{c} · {casos.set_index('Caso').loc[c, 'Visitas']} visitas")

t = d.trayectorias()
serie = t[t["Caso"] == caso].sort_values("Años")
info = casos.set_index("Caso").loc[caso]
k1, k2, k3 = st.columns(3)
k1.metric("Sexo", info["Sexo"])
k2.metric("Edad en la primera visita", f"{info['Edad inicial']:.0f} años")
k3.metric("Seguimiento", f"{info['Seguimiento (años)']:.1f} años")
fig = go.Figure()
fig.add_trace(go.Scatter(x=serie["Años"], y=serie["Diagnóstico"].astype(str), mode="lines", line=dict(shape="hv", color=g.GRIS, width=2),
                         hoverinfo="skip", showlegend=False))
for c in d.CLASES:
    s = serie[serie["Diagnóstico"] == c]
    fig.add_trace(go.Scatter(x=s["Años"], y=s["Diagnóstico"].astype(str), mode="markers", name=c,
                             marker=dict(size=11, color=g.COLOR_CLASE[c], line=dict(width=2, color="white")),
                             hovertemplate="Año %{x:.1f}: %{y}<extra></extra>"))
fig.update_yaxes(type="category", categoryorder="array", categoryarray=d.CLASES, range=[-0.5, 2.5], title=None)
fig.update_xaxes(title="Años desde la primera visita")
st.plotly_chart(g.estilo(fig, alto=280), width="stretch")
