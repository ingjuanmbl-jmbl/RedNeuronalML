# ACA 2 — aplicación de predicción y verificación de artefactos

Esta aplicación Streamlit utiliza el preprocesador serializado y el modelo Keras entrenado para una demostración académica de predicción. Incluye una prueba de integración para verificar su compatibilidad. No reentrena ni modifica los artefactos.

## Requisitos

- Windows de 64 bits
- Python 3.11

## Instalación

Desde la raíz del proyecto, crear el entorno virtual si aún no existe:

```powershell
py -3.11 -m venv .venv
```

Activarlo e instalar las versiones fijadas:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecutar las pruebas

```powershell
python test_model.py
```

La prueba carga `preprocessor_mlp_optimizado.pkl`, verifica el orden de las 13 columnas, aplica `transform()` y exige una matriz densa de 28 características. Luego carga `modelo_mlp_optimizado.keras`, verifica las dimensiones 28→1, ejecuta `predict()` y comprueba que la probabilidad sea finita y esté entre 0 y 1. Finalmente aplica el umbral 0.5.

La observación de prueba toma una categoría existente directamente de cada `categories_` del OneHotEncoder cargado. No redefine ni reemplaza las categorías aprendidas.

## Ejecutar la aplicación Streamlit

Con el entorno virtual activado, desde la raíz del proyecto:

```powershell
streamlit run app.py
```

La aplicación carga el preprocesador y el modelo una sola vez con `st.cache_resource`. Las opciones categóricas se obtienen directamente del OneHotEncoder entrenado y se conservan los nombres y el orden originales de las 13 variables. El formulario muestra una probabilidad y su clase con umbral 0.5.

La interfaz es una demostración académica, no una herramienta de diagnóstico ni un sustituto de la evaluación profesional.

## Limitación conocida de las categorías

No se encontró una copia del CSV ni otro archivo fuente tabular dentro del proyecto para contrastar las etiquetas categóricas con caracteres `�`. Por tanto, se preservan literalmente las categorías almacenadas en el OneHotEncoder. No se debe corregir su texto sin una fuente verificable.
