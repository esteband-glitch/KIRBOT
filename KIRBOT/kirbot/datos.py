"""Carga, limpieza y partición del dataset de intenciones."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATOS_CSV, SEED


def cargar_dataset(ruta: Path | str = DATOS_CSV) -> pd.DataFrame:
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    if not {"texto", "categoria"}.issubset(df.columns):
        raise ValueError("El CSV debe tener las columnas 'texto' y 'categoria'")
    df = df[["texto", "categoria"]].dropna().copy()
    df["texto"] = df["texto"].astype(str).str.strip()
    df["categoria"] = df["categoria"].astype(str).str.strip()
    df = df[(df["texto"] != "") & (df["categoria"] != "")]
    df["_n"] = df["texto"].str.lower().str.replace(r"\s+", " ", regex=True)
    return df.drop_duplicates("_n").drop(columns="_n").reset_index(drop=True)


def dividir(df: pd.DataFrame):
    """Devuelve train, val, test (estratificados) y la lista de etiquetas."""
    enc = LabelEncoder()
    df = df.copy()
    df["label"] = enc.fit_transform(df["categoria"])
    train_val, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
    train, val = train_test_split(train_val, test_size=0.18, random_state=SEED, stratify=train_val["label"])
    r = lambda d: d.reset_index(drop=True)
    return r(train), r(val), r(test), list(enc.classes_)
