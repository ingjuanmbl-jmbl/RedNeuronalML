"""Streamlit demonstration for the trained ACA 2 heart disease MLP."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
PREPROCESSOR_PATH = ROOT / "preprocessor_mlp_optimizado.pkl"
MODEL_PATH = ROOT / "modelo_mlp_optimizado.keras"
THRESHOLD = 0.5

EXPECTED_COLUMNS = [
    "edad", "sexo", "tipo_dolor_toracico", "presion_arterial_reposo",
    "colesterol_serico", "glucemia_ayunas_alta", "electrocardiograma_reposo",
    "frecuencia_cardiaca_max", "angina_inducida_ejercicio", "depresion_ST",
    "pendiente_segmento_ST", "numero_vasos_principales", "resultado_talasemia",
]

NUMERIC_LABELS = {
    "edad": "Edad (años)",
    "presion_arterial_reposo": "Presión arterial en reposo",
    "colesterol_serico": "Colesterol sérico",
    "frecuencia_cardiaca_max": "Frecuencia cardíaca máxima",
    "depresion_ST": "Depresión ST",
}

CATEGORICAL_LABELS = {
    "sexo": "Sexo",
    "tipo_dolor_toracico": "Tipo de dolor torácico",
    "glucemia_ayunas_alta": "Glucemia en ayunas alta",
    "electrocardiograma_reposo": "Electrocardiograma en reposo",
    "angina_inducida_ejercicio": "Angina inducida por ejercicio",
    "pendiente_segmento_ST": "Pendiente del segmento ST",
    "numero_vasos_principales": "Número de vasos principales",
    "resultado_talasemia": "Resultado de talasemia",
}


@st.cache_resource(show_spinner="Cargando el modelo y el preprocesador…")
def load_artifacts() -> tuple[Any, Any]:
    """Load the supplied artifacts once per Streamlit process."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    actual_columns = list(preprocessor.feature_names_in_)
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "El orden de columnas del preprocesador no coincide con el esperado: "
            f"{actual_columns}"
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.input_shape[-1] != 28 or model.output_shape[-1] != 1:
        raise ValueError(
            f"Dimensiones inesperadas: entrada={model.input_shape}, "
            f"salida={model.output_shape}"
        )
    return preprocessor, model


def get_preprocessor_columns(preprocessor: Any) -> tuple[list[str], list[str]]:
    """Read feature groups and learned categories from the fitted transformer."""
    groups = {
        name: list(columns)
        for name, _transformer, columns in preprocessor.transformers_
        if name in {"num", "cat"}
    }
    numeric_columns = groups["num"]
    categorical_columns = groups["cat"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    if len(categorical_columns) != len(encoder.categories_):
        raise ValueError("No coinciden las variables y categorías del OneHotEncoder.")
    return numeric_columns, categorical_columns


def run_inference(
    patient_df: pd.DataFrame, preprocessor: Any, model: Any
) -> tuple[float, int]:
    """Apply the same inference flow as test_model.py."""
    if list(patient_df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Las columnas de entrada no están en el orden esperado.")

    transformed = preprocessor.transform(patient_df)
    if not isinstance(transformed, np.ndarray) or transformed.shape != (1, 28):
        raise ValueError(
            "Se esperaba una matriz densa de forma (1, 28); "
            f"se recibió {getattr(transformed, 'shape', None)}."
        )
    transformed = np.asarray(transformed, dtype=np.float32)

    probabilities = np.asarray(model.predict(transformed, verbose=0)).reshape(-1)
    if probabilities.shape != (1,):
        raise ValueError(f"Forma de predicción inesperada: {probabilities.shape}")
    probability = float(probabilities[0])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probabilidad inválida devuelta por el modelo: {probability}")
    return probability, int(probability >= THRESHOLD)


st.set_page_config(
    page_title="Predicción cardíaca | ACA 2",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px;
             background: linear-gradient(120deg, #eff6ff 0%, #f8fafc 65%, #fff1f2 100%);
             border: 1px solid #e2e8f0; margin-bottom: 1.2rem;}
      .hero p {color: #475569; margin-bottom: 0;}
      .section-caption {color: #64748b; margin-top: -0.45rem; margin-bottom: 1rem;}
      div[data-testid="stForm"] {border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.1rem 1.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>❤️ Estimación de enfermedad cardíaca</h1>
      <p>Demostración académica ACA 2 · Perceptrón multicapa entrenado con el conjunto Cleveland</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Herramienta académica de demostración. Esta predicción no es un diagnóstico "
    "y no sustituye la evaluación de un profesional de la salud."
)

try:
    preprocessor, model = load_artifacts()
    numeric_columns, categorical_columns = get_preprocessor_columns(preprocessor)
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
except Exception as exc:
    st.error(
        "No se pudieron cargar los artefactos. Comprueba las dependencias y que "
        "ambos archivos estén en la raíz del proyecto."
    )
    st.exception(exc)
    st.stop()

st.header("Información clínica y demográfica")
st.markdown(
    '<p class="section-caption">Completa las 13 variables requeridas por el modelo.</p>',
    unsafe_allow_html=True,
)

with st.form("patient_form"):
    st.subheader("Datos demográficos y mediciones")
    number_columns = st.columns(5)
    numeric_values: dict[str, int | float | None] = {}
    for index, column in enumerate(numeric_columns):
        with number_columns[index % len(number_columns)]:
            if column == "depresion_ST":
                numeric_values[column] = st.number_input(
                    NUMERIC_LABELS[column], value=None, step=0.1, format="%.2f",
                    help=f"Variable del modelo: {column}",
                )
            else:
                numeric_values[column] = st.number_input(
                    NUMERIC_LABELS[column], value=None, step=1,
                    help=f"Variable del modelo: {column}",
                )

    st.subheader("Características clínicas categóricas")
    category_columns = st.columns(2)
    categorical_values: dict[str, Any] = {}
    for index, (column, categories) in enumerate(
        zip(categorical_columns, encoder.categories_)
    ):
        with category_columns[index % len(category_columns)]:
            categorical_values[column] = st.selectbox(
                CATEGORICAL_LABELS.get(column, column),
                options=list(categories),
                index=None,
                placeholder="Selecciona una categoría",
                help=f"Categorías aprendidas por el preprocesador · {column}",
            )

    submitted = st.form_submit_button(
        "Calcular estimación", type="primary", use_container_width=True
    )

if submitted:
    missing_fields = [
        column for column in numeric_columns if numeric_values.get(column) is None
    ] + [
        column
        for column in categorical_columns
        if categorical_values.get(column) is None
    ]
    if missing_fields:
        st.error(
            "Completa todos los campos antes de calcular la estimación: "
            + ", ".join(missing_fields)
        )
    else:
        try:
            row_values = {**numeric_values, **categorical_values}
            patient_df = pd.DataFrame(
                [[row_values[column] for column in EXPECTED_COLUMNS]],
                columns=EXPECTED_COLUMNS,
            )
            if list(patient_df.columns) != list(preprocessor.feature_names_in_):
                raise ValueError("Las variables no coinciden con el orden del preprocesador.")
            if not np.isfinite(patient_df[numeric_columns].to_numpy(dtype=float)).all():
                raise ValueError("Los campos numéricos deben contener valores finitos.")

            probability, predicted_class = run_inference(patient_df, preprocessor, model)
            st.header("Resultado de la predicción")
            result_columns = st.columns([1, 1.3])
            with result_columns[0]:
                st.metric("Probabilidad estimada de presencia", f"{probability:.1%}")
                st.progress(probability, text="Probabilidad estimada")
            with result_columns[1]:
                if predicted_class == 1:
                    st.error("Clasificación del modelo: presencia (clase 1)")
                    st.write(
                        f"La probabilidad estimada es igual o superior al umbral "
                        f"de clasificación de {THRESHOLD:.1f}."
                    )
                else:
                    st.success("Clasificación del modelo: ausencia (clase 0)")
                    st.write(
                        f"La probabilidad estimada es inferior al umbral "
                        f"de clasificación de {THRESHOLD:.1f}."
                    )
                st.caption("La clase mostrada corresponde al umbral académico 0.5.")
        except Exception as exc:
            st.error("No fue posible completar la predicción con estos datos.")
            st.exception(exc)

st.divider()
method_column, limitation_column = st.columns(2)
with method_column:
    st.subheader("¿Cómo se calcula?")
    st.write(
        "Las 13 variables se organizan en un DataFrame con los nombres y el orden "
        "originales. El preprocesador aplica sus transformaciones aprendidas y la "
        "red neuronal devuelve una probabilidad. La clase se obtiene comparándola "
        "con el umbral 0.5."
    )
with limitation_column:
    st.subheader("Alcance y limitaciones")
    st.write(
        "El modelo usa un conjunto académico de tamaño limitado y no cuenta con "
        "validación clínica independiente. Sus resultados pueden no generalizar "
        "a otras poblaciones y no deben usarse para tomar decisiones médicas. "
        "Las categorías se muestran tal como están guardadas en el preprocesador."
    )
