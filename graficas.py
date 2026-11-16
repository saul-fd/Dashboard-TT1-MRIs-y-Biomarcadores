"""Módulo "Estructurar paneles de resultados": figuras de Plotly con un mismo estilo para todo el dashboard."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Colores fijos por diagnóstico (los tres primeros de la paleta categórica validada para daltonismo).
# El color sigue a la clase en todas las páginas, aunque un filtro oculte alguna.
COLOR_CLASE = {"CN": "#2a78d6", "MCI": "#eb6834", "AD": "#1baf7a"}
COLOR_SEXO = {"Hombre": "#6250d6", "Mujer": "#e87ba4"}
GRIS = "#9a9893"
AZUL = "#2a78d6"
SECUENCIAL = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
FUENTE = "Source Sans Pro, sans-serif"


def rgba(hex_color: str, alfa: float) -> str:
    r, g_, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g_},{b},{alfa})"


def estilo(fig: go.Figure, alto: int = 380, leyenda: bool = True) -> go.Figure:
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=40, b=10), font=dict(family=FUENTE, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="") if leyenda else None,
        showlegend=leyenda, hoverlabel=dict(font_family=FUENTE),
        bargap=0.25, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


def matriz_confusion(cm: pd.DataFrame, normalizar: bool = True) -> go.Figure:
    """Matriz real (renglones) vs predicho (columnas); el color es el porcentaje de cada renglón."""
    pct = cm.div(cm.sum(1), axis=0)
    texto = [[f"{n}<br>{p:.0%}" for n, p in zip(fila_n, fila_p)] for fila_n, fila_p in zip(cm.values, pct.values)]
    fig = go.Figure(go.Heatmap(
        z=pct.values if normalizar else cm.values, x=[f"Pred. {c}" for c in cm.columns], y=[f"Real {c}" for c in cm.index],
        text=texto, texttemplate="%{text}", colorscale=[[i / 5, c] for i, c in enumerate(SECUENCIAL)],
        zmin=0, zmax=1 if normalizar else None, showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y} → %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return estilo(fig, alto=340, leyenda=False)


def barras_ic(df: pd.DataFrame, x: str, y: str, inf: str, sup: str, color=None, destacar=None, alto=None) -> go.Figure:
    """Barras horizontales con intervalo de confianza; `destacar` pinta en azul las filas indicadas."""
    df = df.sort_values(x)
    colores = color or [AZUL if destacar is not None and d else GRIS for d in (destacar.loc[df.index] if destacar is not None else [False] * len(df))]
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], orientation="h", marker_color=colores, marker_cornerradius=4,
        error_x=dict(type="data", symmetric=False, array=df[sup] - df[x], arrayminus=df[x] - df[inf], color="rgba(110,110,110,.8)", thickness=1.5),
        text=[f"{v:.1%}" for v in df[x]], textposition="inside", insidetextanchor="start", textfont_color="white",
        hovertemplate="%{y}<br>%{x:.1%} (IC 95%: %{customdata[0]:.1%}–%{customdata[1]:.1%})<extra></extra>",
        customdata=np.c_[df[inf], df[sup]],
    ))
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.8, df[sup].max() + 0.05)])
    fig.add_vline(x=1 / 3, line_dash="dot", line_color=GRIS, annotation_text="Azar (33%)", annotation_position="top")
    return estilo(fig, alto=alto or 60 + 34 * len(df), leyenda=False)


def recall_por_clase(df: pd.DataFrame, x: str, alto: int = 360) -> go.Figure:
    """Barras agrupadas de recall por clase; `df` en formato largo con columnas Clase y Recall."""
    fig = px.bar(df, x=x, y="Recall", color="Clase", barmode="group", color_discrete_map=COLOR_CLASE,
                 category_orders={"Clase": list(COLOR_CLASE)}, text_auto=".0%")
    fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
    return estilo(fig, alto=alto)


def mapa_recall(tabla: pd.DataFrame) -> go.Figure:
    """Mapa de calor modelo × clase con el recall; útil cuando hay demasiados modelos para barras agrupadas."""
    fig = go.Figure(go.Heatmap(
        z=tabla.values, x=list(tabla.columns), y=list(tabla.index), text=[[f"{v:.0%}" for v in fila] for fila in tabla.values],
        texttemplate="%{text}", colorscale=[[i / 5, c] for i, c in enumerate(SECUENCIAL)], zmin=0, zmax=1,
        showscale=False, xgap=2, ygap=2, hovertemplate="%{y}<br>Recall de %{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return estilo(fig, alto=60 + 30 * len(tabla), leyenda=False)


def corte(vol: np.ndarray, eje: int, idx: int) -> np.ndarray:
    """Corte 2D orientado para mostrarse con el cerebro derecho (superior arriba, anterior arriba en axial)."""
    s = np.take(vol, idx, axis=eje)
    return np.rot90(s)


def imagen_cerebro(img: np.ndarray, escala="gray", zmin=None, zmax=None, alto=300) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=img[::-1], colorscale=escala, zmin=zmin, zmax=zmax, showscale=escala != "gray",
                               hoverinfo="skip", colorbar=dict(thickness=10, len=0.8)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def imagen_oclusion(img: np.ndarray, calor: np.ndarray, alto=300) -> go.Figure:
    """Corte del cerebro en gris con el mapa de oclusión encima (se ocultan los valores bajos)."""
    fig = imagen_cerebro(img, zmin=0, zmax=255, alto=alto)
    visible = np.where(calor > calor.max() * 0.15, calor, np.nan)[::-1]
    fig.add_trace(go.Heatmap(z=visible, colorscale="Oranges", opacity=0.75, zmin=0, zmax=float(np.nanmax(calor)) or 1,
                             colorbar=dict(thickness=10, len=0.8, title=dict(text="Caída", side="right")),
                             hovertemplate="Caída de la probabilidad: %{z:.2f}<extra></extra>"))
    return fig
