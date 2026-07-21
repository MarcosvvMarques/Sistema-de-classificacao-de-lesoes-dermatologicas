"""
Download and organize the HAM10000 dataset from Kaggle.

Prerequisites:
    1. pip install -r requirements.txt
    2. Copie .env.example para .env e aponte KAGGLE_CONFIG_DIR para a
       pasta que contém kaggle.json

Usage:
    python download_data.py
"""
import json
import os
from pathlib import Path

import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "ham10000"
DATASET  = "kmader/skin-cancer-mnist-ham10000"
EXPECTED_IMAGES = 10_015
REQUIRED_COLUMNS = {
    "lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"
}


def _configure_credentials():
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("Usando KAGGLE_API_TOKEN.")
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("Usando credenciais das variáveis de ambiente.")
        return

    config_dir = Path(
        os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")
    ).expanduser()
    credentials_path = config_dir / "kaggle.json"
    if not credentials_path.is_file():
        raise FileNotFoundError(
            f"Credencial não encontrada: {credentials_path}. "
            "Defina KAGGLE_CONFIG_DIR no arquivo .env."
        )

    try:
        credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"kaggle.json inválido em {credentials_path}") from exc
    missing = {"username", "key"}.difference(credentials)
    if missing:
        raise ValueError(f"kaggle.json não contém os campos: {sorted(missing)}")
    print(f"Usando kaggle.json encontrado em: {credentials_path}")


def _validate_dataset() -> None:
    images = {
        path.stem: path
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for path in DATA_DIR.rglob(pattern)
    }
    metadata_files = sorted(DATA_DIR.rglob("*metadata*.csv"))
    if not metadata_files:
        raise FileNotFoundError("HAM10000_metadata.csv não foi encontrado.")

    frame = pd.read_csv(metadata_files[0])
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Metadados sem as colunas: {sorted(missing_columns)}")
    if len(frame) != EXPECTED_IMAGES:
        raise ValueError(
            f"Esperadas {EXPECTED_IMAGES} linhas de metadados, encontradas {len(frame)}."
        )
    missing_images = set(frame["image_id"]).difference(images)
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} imagens dos metadados não foram encontradas."
        )

    print(f"\nDataset validado: {len(frame)} imagens e 7 classes.")
    print(frame["dx"].value_counts().sort_index())


def main():
    _configure_credentials()
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading HAM10000 from Kaggle dataset: {DATASET}")
    downloaded_path = kagglehub.dataset_download(
        DATASET,
        output_dir=str(DATASET_DIR),
    )
    print(f"Dataset baixado diretamente em: {downloaded_path}")

    _validate_dataset()
    print("\nDownload e validação concluídos.")


if __name__ == "__main__":
    main()
