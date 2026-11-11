"""Compatibility and end-to-end smoke tests for the supplied artifacts."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
PREPROCESSOR_PATH = ROOT / "preprocessor_mlp_optimizado.pkl"
MODEL_PATH = ROOT / "modelo_mlp_optimizado.keras"

EXPECTED_COLUMNS = [
    "edad",
    "sexo",
    "tipo_dolor_toracico",
    "presion_arterial_reposo",
    "colesterol_serico",
    "glucemia_ayunas_alta",
    "electrocardiograma_reposo",
    "frecuencia_cardiaca_max",
    "angina_inducida_ejercicio",
    "depresion_ST",
    "pendiente_segmento_ST",
    "numero_vasos_principales",
    "resultado_talasemia",
]


def build_observation(preprocessor: object) -> pd.DataFrame:
    """Build one row using learned OHE categories, without redefining them."""
    categorical_pipeline = preprocessor.named_transformers_["cat"]
    encoder = categorical_pipeline.named_steps["onehot"]
    learned_categories = encoder.categories_
    categorical_columns = [
        "sexo",
        "tipo_dolor_toracico",
        "glucemia_ayunas_alta",
        "electrocardiograma_reposo",
        "angina_inducida_ejercicio",
        "pendiente_segmento_ST",
        "numero_vasos_principales",
        "resultado_talasemia",
    ]
    values = {
        "edad": 55,
        "presion_arterial_reposo": 130,
        "colesterol_serico": 250,
        "frecuencia_cardiaca_max": 120,
        "depresion_ST": 1.0,
    }
    values.update(
        {
            column: categories[0]
            for column, categories in zip(categorical_columns, learned_categories)
        }
    )
    return pd.DataFrame([[values[column] for column in EXPECTED_COLUMNS]], columns=EXPECTED_COLUMNS)


def main() -> None:
    # Test 1: deserialize and apply the original preprocessor.
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    actual_columns = list(preprocessor.feature_names_in_)
    assert actual_columns == EXPECTED_COLUMNS, (
        f"Preprocessor columns differ: {actual_columns!r}"
    )
    observation = build_observation(preprocessor)
    assert list(observation.columns) == EXPECTED_COLUMNS
    transformed = preprocessor.transform(observation)
    assert isinstance(transformed, np.ndarray), (
        f"Expected a dense numpy array; received {type(transformed).__name__}"
    )
    assert transformed.shape == (1, 28), f"Unexpected shape: {transformed.shape}"
    assert np.issubdtype(transformed.dtype, np.number)
    transformed = np.asarray(transformed, dtype=np.float32)
    print(f"Preprocessor OK: {observation.shape[1]} input columns -> {transformed.shape}")

    # Test 2: load the supplied model and validate its input/output dimensions.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    assert model.input_shape[-1] == 28, f"Unexpected model input: {model.input_shape}"
    assert model.output_shape[-1] == 1, f"Unexpected model output: {model.output_shape}"
    print(f"Model OK: input={model.input_shape}, output={model.output_shape}")

    # Test 3: full DataFrame -> preprocessing -> probability -> class flow.
    probabilities = np.asarray(model.predict(transformed, verbose=0)).reshape(-1)
    assert probabilities.shape == (1,), f"Unexpected prediction shape: {probabilities.shape}"
    probability = float(probabilities[0])
    assert np.isfinite(probability) and 0.0 <= probability <= 1.0, (
        f"Invalid probability: {probability}"
    )
    predicted_class = int(probability >= 0.5)
    print(
        "Integration OK: "
        f"probability={probability:.8f}, threshold=0.5, class={predicted_class}"
    )


if __name__ == "__main__":
    main()
