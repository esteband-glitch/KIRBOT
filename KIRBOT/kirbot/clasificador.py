"""Clasificadores de intención: Transformer (DistilBERT) y TF-IDF + regresión logística."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (CARPETA_MODELOS, MAX_LONGITUD, MODELO_BASE, RUTA_TFIDF, RUTA_TRANSFORMER, SEED)


@dataclass
class Prediccion:
    categoria: str
    confianza: float
    probabilidades: dict[str, float] = field(default_factory=dict)
    fuente: str = "modelo"  # modelo | reglas | modelo+reglas


class Clasificador:
    nombre = "base"
    umbral_confianza = 0.40
    etiquetas: list[str] = []

    def predecir_lote(self, textos: list[str]) -> list[Prediccion]:  # pragma: no cover
        raise NotImplementedError

    def predecir(self, texto: str) -> Prediccion:
        return self.predecir_lote([texto])[0]

    @staticmethod
    def _a_prediccion(probs: np.ndarray, etiquetas: list[str]) -> Prediccion:
        i = int(np.argmax(probs))
        return Prediccion(etiquetas[i], float(probs[i]), {e: float(p) for e, p in zip(etiquetas, probs)})


# ── TF-IDF (rápido, sin GPU: ideal para probar el diálogo) ───────────────────
class ClasificadorTfidf(Clasificador):
    nombre = "tfidf"
    umbral_confianza = 0.30

    def __init__(self, pipeline, etiquetas: list[str]):
        self.pipeline, self.etiquetas = pipeline, etiquetas

    @classmethod
    def entrenar(cls, textos, etiquetas_texto: list[str]) -> "ClasificadorTfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        etiquetas = sorted(set(etiquetas_texto))
        y = [etiquetas.index(e) for e in etiquetas_texto]
        pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("palabras", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
                ("letras", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), strip_accents="unicode",
                                           sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=20, max_iter=3000, random_state=SEED)),
        ])
        pipe.fit(list(textos), y)
        return cls(pipe, etiquetas)

    def predecir_lote(self, textos):
        probs = self.pipeline.predict_proba(list(textos))
        return [self._a_prediccion(p, self.etiquetas) for p in probs]

    def guardar(self, ruta: Path = RUTA_TFIDF):
        import joblib
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "etiquetas": self.etiquetas}, ruta)

    @classmethod
    def cargar(cls, ruta: Path = RUTA_TFIDF):
        import joblib
        d = joblib.load(ruta)
        return cls(d["pipeline"], d["etiquetas"])


# ── Transformer ──────────────────────────────────────────────────────────────
class ClasificadorTransformer(Clasificador):
    nombre = "transformer"
    umbral_confianza = 0.40

    def __init__(self, modelo, tokenizer, etiquetas: list[str]):
        import torch
        self.torch = torch
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo = modelo.to(self.dispositivo).eval()
        self.tokenizer, self.etiquetas = tokenizer, etiquetas

    def predecir_lote(self, textos):
        torch = self.torch
        salida = []
        for i in range(0, len(textos), 32):
            lote = list(textos[i:i + 32])
            ent = self.tokenizer(lote, return_tensors="pt", truncation=True, max_length=MAX_LONGITUD, padding=True)
            ent = {k: v.to(self.dispositivo) for k, v in ent.items()}
            with torch.no_grad():
                probs = torch.softmax(self.modelo(**ent).logits, dim=-1).cpu().numpy()
            salida += [self._a_prediccion(p, self.etiquetas) for p in probs]
        return salida

    @classmethod
    def cargar(cls, ruta: Path = RUTA_TRANSFORMER):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ruta))
        mod = AutoModelForSequenceClassification.from_pretrained(str(ruta))
        etiquetas = json.loads((Path(ruta) / "etiquetas.json").read_text(encoding="utf-8"))
        return cls(mod, tok, etiquetas)


def entrenar_transformer(train_df, val_df, etiquetas: list[str], epocas: int = 10, lr: float = 5e-5,
                         guardar_en: Path | None = RUTA_TRANSFORMER):
    """Fine-tuning de DistilBERT multilingüe. Devuelve (ClasificadorTransformer, historial_de_logs)."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)
    from sklearn.metrics import accuracy_score, f1_score

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODELO_BASE)

    class _Datos(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tok(list(df["texto"]), truncation=True, max_length=MAX_LONGITUD)
            self.y = [int(v) for v in df["label"]]

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    def metricas(pred):
        est = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(pred.label_ids, est),
                "f1_macro": f1_score(pred.label_ids, est, average="macro", zero_division=0)}

    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE, num_labels=len(etiquetas),
        id2label=dict(enumerate(etiquetas)), label2id={e: i for i, e in enumerate(etiquetas)})

    args = TrainingArguments(
        output_dir=str(CARPETA_MODELOS / "_checkpoints"),
        num_train_epochs=epocas, learning_rate=lr, weight_decay=0.01, warmup_ratio=0.1,
        per_device_train_batch_size=8, per_device_eval_batch_size=16,
        eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
        report_to="none", seed=SEED, fp16=torch.cuda.is_available(), dataloader_num_workers=0)

    trainer = Trainer(model=modelo, args=args, train_dataset=_Datos(train_df), eval_dataset=_Datos(val_df),
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metricas)
    trainer.train()

    if guardar_en:
        guardar_en = Path(guardar_en)
        guardar_en.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(guardar_en))
        tok.save_pretrained(str(guardar_en))
        (guardar_en / "etiquetas.json").write_text(json.dumps(etiquetas, ensure_ascii=False), encoding="utf-8")
    return ClasificadorTransformer(trainer.model, tok, etiquetas), trainer.state.log_history


# ── Utilidades ───────────────────────────────────────────────────────────────
def cargar_clasificador(preferido: str = "auto") -> Clasificador:
    """auto: usa el Transformer entrenado si existe; si no, TF-IDF (lo entrena en ~1 s si falta)."""
    if preferido in ("auto", "transformer") and (RUTA_TRANSFORMER / "etiquetas.json").exists():
        try:
            return ClasificadorTransformer.cargar()
        except Exception as e:  # sin torch/transformers instalados, etc.
            if preferido == "transformer":
                raise
            print(f"[KIRBOT] No pude cargar el Transformer ({e}). Uso TF-IDF.")
    elif preferido == "transformer":
        raise FileNotFoundError("Aún no hay Transformer entrenado. Ejecuta: python entrenar.py --modelo transformer")

    if RUTA_TFIDF.exists():
        try:
            return ClasificadorTfidf.cargar()
        except Exception:
            pass
    from .datos import cargar_dataset
    df = cargar_dataset()
    print("[KIRBOT] Entrenando clasificador TF-IDF de respaldo…")
    clf = ClasificadorTfidf.entrenar(df["texto"], list(df["categoria"]))
    clf.guardar()
    return clf


def evaluar(clasificador: Clasificador, df, etiquetas: list[str] | None = None, titulo: str = "",
            guardar_matriz: Path | None = None, mostrar: bool = False):
    """Imprime reporte de clasificación y (opcional) guarda/muestra la matriz de confusión."""
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
    etiquetas = etiquetas or clasificador.etiquetas
    pred = [p.categoria for p in clasificador.predecir_lote(list(df["texto"]))]
    real = list(df["categoria"])
    print(f"== {titulo} | accuracy = {accuracy_score(real, pred):.3f} ==")
    print(classification_report(real, pred, labels=etiquetas, zero_division=0))
    if guardar_matriz or mostrar:
        import matplotlib
        if not mostrar:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(real, pred, labels=etiquetas, xticks_rotation=45, ax=ax,
                                                colorbar=False)
        ax.set_title(f"Matriz de confusión – {titulo}")
        fig.tight_layout()
        if guardar_matriz:
            Path(guardar_matriz).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(guardar_matriz, dpi=120)
        if mostrar:
            plt.show()
        plt.close(fig)
    errores = df.assign(prediccion=pred)
    return errores[errores["categoria"] != errores["prediccion"]]
