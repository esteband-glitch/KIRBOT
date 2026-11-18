"""Entrena y evalúa los clasificadores de KIRBOT.

  python entrenar.py --modelo tfidf            # ~1 s, sin GPU (para pruebas rápidas)
  python entrenar.py --modelo transformer      # DistilBERT multilingüe, 10 épocas (CPU: unos minutos)
  python entrenar.py --modelo ambos --epocas 15
"""
import argparse

import pandas as pd

from kirbot.clasificador import ClasificadorTfidf, entrenar_transformer, evaluar
from kirbot.config import CARPETA_REPORTES, DATOS_CSV, RUTA_TFIDF, RUTA_TRANSFORMER
from kirbot.datos import cargar_dataset, dividir


def graficar_perdida(log_history, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reg = pd.DataFrame(log_history)
    plt.figure(figsize=(8, 4))
    for campo, nombre in [("loss", "Entrenamiento"), ("eval_loss", "Validación")]:
        if campo in reg.columns:
            d = reg.dropna(subset=[campo])
            plt.plot(d["epoch"], d[campo], marker="o", label=nombre)
    plt.title("Pérdida durante el entrenamiento de KIRBOT")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=.2); plt.tight_layout()
    plt.savefig(ruta, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["tfidf", "transformer", "ambos"], default="tfidf")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--datos", default=str(DATOS_CSV), help="CSV con columnas texto,categoria")
    a = ap.parse_args()

    df = cargar_dataset(a.datos)
    train, val, test, etiquetas = dividir(df)
    print(f"Datos: {len(df)} | train {len(train)} · val {len(val)} · test {len(test)} | clases: {etiquetas}")
    print(df["categoria"].value_counts().to_string(), "\n")
    CARPETA_REPORTES.mkdir(exist_ok=True)

    if a.modelo in ("tfidf", "ambos"):
        clf = ClasificadorTfidf.entrenar(train["texto"], list(train["categoria"]))
        evaluar(clf, val, titulo="TF-IDF · validación")
        err = evaluar(clf, test, titulo="TF-IDF · prueba", guardar_matriz=CARPETA_REPORTES / "matriz_tfidf.png")
        # el modelo final de respaldo se reentrena con TODOS los datos
        ClasificadorTfidf.entrenar(df["texto"], list(df["categoria"])).guardar()
        print("Guardado:", RUTA_TFIDF, "\nErrores en prueba:\n", err[["texto", "categoria", "prediccion"]].to_string(), "\n")

    if a.modelo in ("transformer", "ambos"):
        clf, log = entrenar_transformer(train, val, etiquetas, epocas=a.epocas)
        graficar_perdida(log, CARPETA_REPORTES / "perdida_transformer.png")
        evaluar(clf, val, titulo="Transformer · validación")
        err = evaluar(clf, test, titulo="Transformer · prueba",
                      guardar_matriz=CARPETA_REPORTES / "matriz_transformer.png")
        print("Guardado:", RUTA_TRANSFORMER, "\nErrores en prueba:\n", err[["texto", "categoria", "prediccion"]].to_string())


if __name__ == "__main__":
    main()
