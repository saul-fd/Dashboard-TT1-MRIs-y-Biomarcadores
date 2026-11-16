# Dashboard TT1 · MRI y biomarcadores

Dashboard en Streamlit del Trabajo Terminal.

- La página **Resumen** presenta la descripción del proyecto, métricas, gráficas de desempeño y espacios para hallazgos.
- Las demás páginas describen en texto el contenido previsto.
- Las métricas y gráficas se calculan a partir de datos aleatorios con semilla fija para mantener la consistencia entre ejecuciones.

## Ejecutar

Con Python 3.10 o superior, desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Ramas

- `main`: resumen con métricas y gráficas, y descripciones de las demás secciones.
- `dashboard-avanzado`: versión previa con datos sintéticos, gráficas y simulación de resultados.
