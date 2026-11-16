"""Datos aleatorios y metricas del resumen, generados con semilla fija."""


import numpy as np
import pandas as pd
import streamlit as st


SEED = 42


CLASES = ["CN", "MCI", "AD"]


GRUPOS_EDAD = [50, 65, 70, 75, 80, 85, 100]


ETIQUETAS_EDAD = ["<65", "65–69", "70–74", "75–79", "80–84", "85+"]


GRUPOS_IMC = [0, 18.5, 25, 30, 100]


ETIQUETAS_IMC = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]


MODELO_MRI = "DenseNet-121 3D"


MODELO_PLASMA = "XGBoost (plasma)"


def grupo_edad(edad: pd.Series) -> pd.Series:
    return pd.cut(edad, GRUPOS_EDAD, labels=ETIQUETAS_EDAD, right=False)


def _diagnosticos(rng, n: int) -> np.ndarray:
    return rng.choice(3, size=n, p=[0.40, 0.40, 0.20])


@st.cache_data
def cohorte(n: int = 1200) -> pd.DataFrame:
    """Un renglón por sujeto ficticio."""
    rng = np.random.default_rng(SEED)
    y = _diagnosticos(rng, n)
    edad = np.clip(rng.normal(71 + 2.5 * y, 7), 55, 90).round()
    imc = np.clip(rng.normal(27 - 0.8 * y, 4.5), 16, 45).round(1)
    conjunto = rng.choice(["Pareado con plasma", "Solo MRI"], size=n, p=[0.35, 0.65])
    particion = np.where(conjunto == "Solo MRI", "Solo entrenamiento",
                         rng.choice(["Entrenamiento/validación", "Prueba"], size=n, p=[0.8, 0.2]))
    return pd.DataFrame({
        "Diagnóstico": pd.Categorical(np.array(CLASES)[y], CLASES),
        "Conjunto": conjunto,
        "Partición": particion,
        "Sexo": rng.choice(["Mujer", "Hombre"], size=n),
        "Edad": edad,
        "Grupo de edad": grupo_edad(pd.Series(edad)),
        "IMC": imc,
        "Rango de IMC": pd.cut(imc, GRUPOS_IMC, labels=ETIQUETAS_IMC, right=False),
    })


def _simular(rng, y: np.ndarray, separacion: list) -> np.ndarray:
    """Probabilidades ficticias: logits aleatorios con un empuje hacia la clase real."""
    logits = rng.normal(0, 1, (len(y), 3))
    logits[np.arange(len(y)), y] += np.array(separacion)[y]
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


@st.cache_data
def predicciones() -> dict:
    """{modelo: DataFrame con Group, particion, p_CN, p_MCI, p_AD}; MRI y plasma sobre los mismos sujetos ficticios."""
    rng = np.random.default_rng(SEED + 4)
    particion = np.r_[np.repeat("cv", 600), np.repeat("test", 170)]
    y = _diagnosticos(rng, len(particion))
    base = pd.DataFrame({"Group": np.array(CLASES)[y], "particion": particion})
    cols = [f"p_{c}" for c in CLASES]
    mri = _simular(rng, y, [1.6, 0.9, 1.8])
    plasma = _simular(rng, y, [1.8, 0.4, 2.3])
    salida = {MODELO_MRI: mri, MODELO_PLASMA: plasma}
    for w in (0.3, 0.5, 0.7):
        salida[f"Fusión tardía {w:.1f}·plasma + {1 - w:.1f}·MRI"] = w * plasma + (1 - w) * mri
    salida["Apilamiento plasma + MRI"] = _simular(rng, y, [1.9, 0.8, 2.3])
    return {k: base.assign(**dict(zip(cols, v.T))) for k, v in salida.items()}


def y_prob(pred: pd.DataFrame, particion: str = "test", peso_mci: float = 1.0):
    """Etiquetas reales, probabilidades y clase predicha (con la probabilidad de MCI multiplicada por `peso_mci`)."""
    p = pred[pred["particion"] == particion]
    y = p["Group"].map({c: i for i, c in enumerate(CLASES)}).values
    prob = p[[f"p_{c}" for c in CLASES]].values
    return y, prob, (prob * np.array([1.0, peso_mci, 1.0])).argmax(1)


@st.cache_data
def ba_con_ic(y, p, n_boot: int = 1000):
    """Bootstrap vectorizado: evita miles de llamadas a un evaluador externo."""
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(y), size=(n_boot, len(y)))
    etiquetas = y[indices]
    aciertos = (y == p)[indices]
    recalls = []
    for clase in np.unique(y):
        mascara = etiquetas == clase
        total = mascara.sum(axis=1)
        recalls.append(np.divide((mascara & aciertos).sum(axis=1), total,
                                 out=np.full(n_boot, np.nan), where=total > 0))
    vals = np.nanmean(recalls, axis=0)
    ba = np.mean([(p[y == clase] == clase).mean() for clase in np.unique(y)])
    return ba, *np.percentile(vals, [2.5, 97.5])


@st.cache_data
def tabla_modelos() -> pd.DataFrame:
    """Métricas en prueba de cada modelo de `predicciones()`."""
    filas = []
    for nombre, pred in predicciones().items():
        y, prob, p = y_prob(pred)
        ba, inf, sup = ba_con_ic(y, p)
        rec = {clase: (p[y == k] == k).mean() for k, clase in enumerate(CLASES)}
        familia = "Fusión" if nombre.startswith(("Fusión", "Apilamiento")) else "Un solo tipo de dato"
        filas.append({"modelo": nombre, "familia": familia, "test": ba, "ic_inf": inf, "ic_sup": sup,
                      **{f"recall_{c}": rec[c] for c in CLASES}})
    return pd.DataFrame(filas)
