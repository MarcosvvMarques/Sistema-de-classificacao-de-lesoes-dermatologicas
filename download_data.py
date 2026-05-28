"""
Download and organize the HAM10000 dataset from Kaggle.

Prerequisites:
    1. pip install -r requirements.txt
    2. Copie .env.example para .env e preencha com suas credenciais Kaggle

Usage:
    python download_data.py
"""
import os
import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
DATASET  = "kmader/skin-cancer-mnist-ham10000"


def _configure_credentials():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USERNAME"]
        os.environ["KAGGLE_KEY"]      = os.environ["KAGGLE_KEY"]
        print("Usando credenciais das variáveis de ambiente.")
    else:
        print("Usando ~/.kaggle/kaggle.json.")


def main():
    _configure_credentials()
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading HAM10000 from Kaggle dataset: {DATASET}")
    cached_path = kagglehub.dataset_download(DATASET)
    cached_path = Path(cached_path)
    print(f"Dataset baixado em: {cached_path}")

    # Copia arquivos do cache do kagglehub para data/
    for item in cached_path.rglob("*"):
        if item.is_file():
            dest = DATA_DIR / item.relative_to(cached_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(item, dest)

    # Garante diretórios de imagens
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        (DATA_DIR / part).mkdir(exist_ok=True)

    # Verifica metadata CSV
    meta = next(DATA_DIR.rglob("*metadata*.csv"), None)
    if meta:
        import pandas as pd
        df = pd.read_csv(meta)
        print(f"\nMetadata loaded: {len(df)} rows, columns: {list(df.columns)}")
        print(df["dx"].value_counts())
    else:
        print(f"\nMetadata CSV não encontrado em {DATA_DIR}. Verifique os arquivos extraídos.")

    print("\nDownload completo.")


if __name__ == "__main__":
    main()
