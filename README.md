# ⚽ FutbolData

Este repositorio contiene scripts en **Python** para analizar y visualizar datos de fútbol (tracking y eventos) usando los datasets de **sample-data-master**.

## 🚀 Requisitos

- [Python 3.12+](https://www.python.org/downloads/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- (opcional) [Jupyter Notebook](https://jupyter.org/)

Instalar dependencias necesarias:

```bash
pip install pandas matplotlib
```

# ▶️ Cómo ejecutar los scripts

⚠️ Importante: Los scripts usan rutas relativas como data/Sample_Game_1/..., por lo que es obligatorio estar dentro de la carpeta sample-data-master antes de ejecutarlos.
Si no, aparecerán errores FileNotFoundError.

Pasos para ejecutar:

Clona el repositorio y entra en la carpeta principal:

git clone https://github.com/danielgarciaN/BigData.git
cd BigData


Accede a la carpeta donde están los scripts:

```bash
cd sample-data-master
```

Ejecuta el script que quieras. Ejemplos:

# Ejecutar el script de mapa de calor
python heatmap_fast.py

# Ejecutar el script de grafo de pases
python pass_graph.py

# Ejecutar el script de posesión
python possesion_pie.py

# Ejecutar el script de valoración de jugadores
python rating.py
