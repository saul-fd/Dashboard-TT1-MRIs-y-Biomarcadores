"""Módulo "Cargar resultados" del dashboard (versión mockup).

Todos los datos son FICTICIOS: se generan con una semilla fija para mostrar la estructura de los paneles.
No se lee ningún archivo de ADNI ni de resultados; cuando los subsistemas entreguen sus productos, cada
función se sustituye por la lectura correspondiente sin cambiar su firma.
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

SEED = 42
CLASES = ["CN", "MCI", "AD"]
GRUPOS_EDAD = [50, 65, 70, 75, 80, 85, 100]
ETIQUETAS_EDAD = ["<65", "65–69", "70–74", "75–79", "80–84", "85+"]
GRUPOS_IMC = [0, 18.5, 25, 30, 100]
ETIQUETAS_IMC = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]

MODELO_MRI = "DenseNet-121 3D"
MODELO_PLASMA = "XGBoost (plasma)"
PESO_MCI_FINAL = 1.5

BIOMARCADORES = {
    "pT217": "p-tau217 (pg/mL)",
    "AB42_AB40": "Aβ42/Aβ40",
    "pT217_AB42": "p-tau217/Aβ42",
    "NfL": "NfL (pg/mL)",
    "GFAP": "GFAP (pg/mL)",
}


def grupo_edad(edad: pd.Series) -> pd.Series:
    return pd.cut(edad, GRUPOS_EDAD, labels=ETIQUETAS_EDAD, right=False)


def _diagnosticos(rng, n: int) -> np.ndarray:
    return rng.choice(3, size=n, p=[0.40, 0.40, 0.20])


# ---------------------------------------------------------------- cohorte y biomarcadores

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


@st.cache_data
def biomarcadores(n: int = 1000) -> pd.DataFrame:
    """Muestras de plasma ficticias; cada biomarcador se desplaza con el diagnóstico."""
    rng = np.random.default_rng(SEED + 1)
    y = _diagnosticos(rng, n)
    edad = np.clip(rng.normal(71 + 2.5 * y, 7), 55, 90).round()
    pt217 = np.exp(rng.normal(-1.8 + 0.5 * y + 0.01 * (edad - 70), 0.5))
    ab = np.clip(rng.normal(0.095 - 0.008 * y, 0.012), 0.04, None)
    return pd.DataFrame({
        "pT217": pt217,
        "AB42_AB40": ab,
        "pT217_AB42": pt217 / (ab * 120),
        "NfL": np.exp(rng.normal(2.9 + 0.2 * y + 0.015 * (edad - 70), 0.4)),
        "GFAP": np.exp(rng.normal(4.4 + 0.25 * y + 0.01 * (edad - 70), 0.5)),
        "Diagnóstico": pd.Categorical(np.array(CLASES)[y], CLASES),
        "Sexo": rng.choice(["Mujer", "Hombre"], size=n),
        "Edad": edad,
        "Grupo de edad": grupo_edad(pd.Series(edad)),
        "Lote": rng.choice(["Lote A", "Lote B", "Lote C"], size=n),
    })


@st.cache_data
def poder_discriminante() -> pd.DataFrame:
    """AUC univariada de cada biomarcador para separar cada clase del resto (orientada para que sea >= 0.5)."""
    b = biomarcadores()
    filas = []
    for col, nombre in BIOMARCADORES.items():
        for c in CLASES:
            auc = roc_auc_score(b["Diagnóstico"] == c, b[col])
            filas.append({"Biomarcador": nombre, "Clase": c, "AUC": max(auc, 1 - auc)})
    return pd.DataFrame(filas)


@st.cache_data
def importancia_variables() -> pd.DataFrame:
    """Importancia ficticia de las variables del modelo de plasma."""
    variables = ["p-tau217", "p-tau217/Aβ42", "Aβ42/Aβ40", "GFAP", "NfL", "Edad", "Escolaridad", "Sexo", "IMC"]
    imp = np.sort(np.random.default_rng(SEED + 2).dirichlet(np.linspace(6, 1, len(variables))))[::-1]
    return pd.DataFrame({"Variable": variables, "Importancia": imp})


# ---------------------------------------------------------------- demográficos (estructura de Diagnosis_demograficos.csv)

FASES = ["ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4"]
# Variables categóricas de PTDEMOG con sus etiquetas (códigos del diccionario de ADNI) y probabilidades ficticias
VARIABLES_DEMOGRAFICAS = {
    "Estado civil": ({"Casado(a)": 0.70, "Viudo(a)": 0.11, "Divorciado(a)": 0.10, "Nunca casado(a)": 0.05, "Unión libre": 0.04}, "PTMARRY"),
    "Tipo de residencia": ({"Casa propia": 0.72, "Condominio propio": 0.10, "Departamento rentado": 0.07, "Comunidad de retiro": 0.05,
                            "Casa rentada": 0.04, "Otro": 0.02}, "PTHOME"),
    "Jubilado(a)": ({"Sí": 0.77, "No": 0.22, "No aplica": 0.01}, "PTNOTRT"),
    "Dominancia manual": ({"Diestro(a)": 0.90, "Zurdo(a)": 0.10}, "PTHAND"),
    "Idioma principal": ({"Inglés": 0.95, "Español": 0.03, "Otro": 0.02}, "PTPLANG"),
    "Etnicidad": ({"No hispano o latino": 0.93, "Hispano o latino": 0.06, "Desconocido": 0.01}, "PTETHCAT"),
    "Raza": ({"Blanca": 0.85, "Negra o afroamericana": 0.08, "Asiática": 0.04, "Más de una": 0.02, "Desconocida": 0.01}, "PTRACCAT"),
}


@st.cache_data
def demograficos(n: int = 900) -> pd.DataFrame:
    """Un renglón por visita con diagnóstico; los datos demográficos son fijos por participante (todo ficticio)."""
    rng = np.random.default_rng(SEED + 7)
    fase = rng.choice(FASES, size=n, p=[0.25, 0.05, 0.30, 0.22, 0.18])
    inicio = {"ADNI1": 2005, "ADNIGO": 2009, "ADNI2": 2011, "ADNI3": 2016, "ADNI4": 2022}
    y0 = rng.choice(3, size=n, p=[0.40, 0.42, 0.18])
    base = pd.DataFrame({
        "Participante": np.arange(n),
        "Fase ADNI": fase,
        "Sexo": rng.choice(["Mujer", "Hombre"], size=n),
        "Escolaridad (años)": np.clip(rng.normal(16 - 0.4 * y0, 2.6), 6, 20).round(),
        "Edad inicial": np.clip(rng.normal(71 + 2 * y0, 7), 55, 90).round(),
        "Año inicial": [inicio[f] + int(rng.integers(0, 3)) for f in fase],
        "Inicio de síntomas": np.where(y0 > 0, -rng.gamma(2.0, 1.5 + y0, n), np.nan),
        "dx0": y0,
        **{var: rng.choice(list(p), size=n, p=list(p.values())) for var, (p, _) in VARIABLES_DEMOGRAFICAS.items()},
    })
    filas = []
    for pid, anio, estado in zip(base["Participante"], base["Año inicial"], base["dx0"]):
        for v in range(min(int(rng.integers(1, 9)), 2026 - anio + 1)):
            if v and estado < 2 and rng.random() < (0.07, 0.2)[estado]:
                estado += 1
            filas.append((pid, v, anio + v, estado))
    visitas = pd.DataFrame(filas, columns=["Participante", "Visita", "Año del examen", "dx"])
    df = visitas.merge(base, on="Participante")
    df["Edad"] = df["Edad inicial"] + df["Visita"]
    df["Grupo de edad"] = grupo_edad(df["Edad"])
    df["Años desde inicio de síntomas"] = (df["Visita"] - df["Inicio de síntomas"]).round(1)
    df["Diagnóstico"] = pd.Categorical(np.array(CLASES)[df["dx"]], CLASES)
    return df.drop(columns=["dx", "dx0", "Año inicial", "Inicio de síntomas"])


# ---------------------------------------------------------------- evolución longitudinal

@st.cache_data
def trayectorias(n: int = 500) -> pd.DataFrame:
    """Diagnóstico de casos ficticios en cada visita; progresan como una cadena de Markov CN -> MCI -> AD."""
    rng = np.random.default_rng(SEED + 3)
    paso = {0: 0.06, 1: 0.18, 2: 0.0}  # probabilidad anual de pasar a la siguiente etapa
    filas = []
    for i in range(1, n + 1):
        estado = rng.choice(3, p=[0.5, 0.35, 0.15])
        edad = int(np.clip(rng.normal(70 + 2 * estado, 7), 55, 88))
        sexo = rng.choice(["Mujer", "Hombre"])
        visitas = rng.integers(2, 12)
        anios = np.cumsum(np.r_[0, rng.uniform(0.5, 2.0, visitas - 1)])
        for t0, t1 in zip(np.r_[0, anios[:-1]], anios):
            if rng.random() < paso[estado] * (t1 - t0) * (1 + 0.04 * (edad - 70)):
                estado += 1
            filas.append({"Caso": f"Caso {i:04d}", "Años": round(float(t1), 2), "Diagnóstico": CLASES[estado],
                          "Sexo": sexo, "Edad inicial": edad})
    t = pd.DataFrame(filas)
    t["Diagnóstico"] = pd.Categorical(t["Diagnóstico"], CLASES)
    return t


@st.cache_data
def transiciones() -> pd.DataFrame:
    """Primer y último diagnóstico de cada caso."""
    t = trayectorias()
    g = t.groupby("Caso", observed=True)
    res = pd.DataFrame({
        "Inicial": g["Diagnóstico"].first(),
        "Final": g["Diagnóstico"].last(),
        "Seguimiento (años)": g["Años"].max(),
        "Visitas": g.size(),
        "Sexo": g["Sexo"].first(),
        "Edad inicial": g["Edad inicial"].first(),
    }).reset_index()
    res["Grupo de edad"] = grupo_edad(res["Edad inicial"])
    orden = {c: i for i, c in enumerate(CLASES)}
    res["Progresó"] = res["Final"].map(orden).astype(int) > res["Inicial"].map(orden).astype(int)
    return res


# ---------------------------------------------------------------- predicciones y métricas

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


def matriz_confusion(y, p) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y, p, labels=range(3)), index=CLASES, columns=CLASES)


def reporte_por_clase(y, prob, p) -> pd.DataFrame:
    """Precision, recall, F1 y PR-AUC por clase (requisito RF-07)."""
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=range(3), zero_division=0)
    pr_auc = [average_precision_score(y == k, prob[:, k]) for k in range(3)]
    roc = [roc_auc_score(y == k, prob[:, k]) for k in range(3)]
    return pd.DataFrame({"Precision": pr, "Recall": rc, "F1": f1, "PR-AUC": pr_auc, "ROC-AUC": roc, "Sujetos": n}, index=CLASES)


@st.cache_data
def ba_con_ic(y, p, n_boot: int = 1000):
    """Exactitud balanceada con intervalo de confianza al 95% por bootstrap."""
    rng = np.random.default_rng(SEED)
    vals = [balanced_accuracy_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return balanced_accuracy_score(y, p), *np.percentile(vals, [2.5, 97.5])


@st.cache_data
def tabla_modelos() -> pd.DataFrame:
    """Métricas en prueba de cada modelo de `predicciones()`."""
    filas = []
    for nombre, pred in predicciones().items():
        y, prob, p = y_prob(pred)
        ba, inf, sup = ba_con_ic(y, p)
        rec = reporte_por_clase(y, prob, p)["Recall"]
        familia = "Fusión" if nombre.startswith(("Fusión", "Apilamiento")) else "Un solo tipo de dato"
        filas.append({"modelo": nombre, "familia": familia, "test": ba, "ic_inf": inf, "ic_sup": sup,
                      **{f"recall_{c}": rec[c] for c in CLASES}})
    return pd.DataFrame(filas)


@st.cache_data
def historial(epocas: int = 60) -> pd.DataFrame:
    """Curva de entrenamiento ficticia de la DenseNet-121 (promedio de folds)."""
    rng = np.random.default_rng(SEED + 5)
    e = np.arange(1, epocas + 1)
    return pd.DataFrame({"epoch": e, "train_loss": 1.1 * np.exp(-e / 25) + 0.35 + rng.normal(0, 0.02, epocas),
                         "val_bal_acc": 0.33 + 0.27 * (1 - np.exp(-e / 12)) + rng.normal(0, 0.015, epocas)})


# ---------------------------------------------------------------- imágenes

FORMA = (80, 96, 80)  # sagital × coronal × axial, como los volúmenes a 2 mm


def _malla():
    return np.meshgrid(*(np.linspace(-1, 1, n) for n in FORMA), indexing="ij")


@st.cache_data
def cerebros_promedio() -> dict:
    """Cerebro sintético (fantoma) por diagnóstico: ventrículos más grandes y corteza más delgada al avanzar la enfermedad."""
    x, y, z = _malla()
    textura = gaussian_filter(np.random.default_rng(SEED + 6).normal(0, 1, FORMA), 2.5)
    out = {}
    for k, c in enumerate(CLASES):
        r = np.sqrt((x / 0.85) ** 2 + (y / 0.9) ** 2 + ((z - 0.05) / 0.8) ** 2)
        vol = np.where(r < 1, 120.0, 0.0)
        vol[r < 0.82 - 0.03 * k] = 205
        vol[(r > 0.86) & (r < 1) & (textura > 0.25 - 0.12 * k)] = 50  # surcos
        esc = 1 + 0.35 * k
        for lado in (-1, 1):
            vent = np.sqrt(((x - 0.12 * lado) / (0.07 * esc)) ** 2 + ((y + 0.05) / (0.35 * esc ** 0.5)) ** 2 + ((z - 0.08) / (0.12 * esc)) ** 2)
            vol[vent < 1] = 25
        out[c] = gaussian_filter(vol, 1.0)
    return out


@st.cache_data
def mapa_oclusion(clase: str) -> np.ndarray:
    """Mapa de oclusión ficticio en cubos de 8 vóxeles (caída de la probabilidad de la clase)."""
    x, y, z = _malla()
    centros = {"AD": [(-0.45, 0.05, -0.2), (0.45, 0.05, -0.2), (0.0, -0.05, 0.1)], "MCI": [(-0.45, 0.1, -0.25), (0.3, -0.4, 0.2)]}
    calor = sum(np.exp(-(((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2) / 0.03)) for a, b, c in centros.get(clase, []))
    bloques = calor[::8, ::8, ::8]
    return np.kron(bloques, np.ones((8, 8, 8)))[: FORMA[0], : FORMA[1], : FORMA[2]] * (0.14 if clase == "AD" else 0.1)
