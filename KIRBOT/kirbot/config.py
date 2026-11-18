"""Configuración central. Edita aquí los datos del negocio y las rutas."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS_CSV = RAIZ / "data" / "ejemplos.csv"
CARPETA_MODELOS = RAIZ / "modelos"
RUTA_TRANSFORMER = CARPETA_MODELOS / "kirbot_transformer"
RUTA_TFIDF = CARPETA_MODELOS / "kirbot_tfidf.joblib"
CARPETA_REPORTES = RAIZ / "reportes"

SEED = 42
MODELO_BASE = "distilbert-base-multilingual-cased"
MAX_LONGITUD = 80

# Velocidad de la simulación del estado de los pedidos (1 = tiempo real).
# Para demos rápidas: KIRBOT_VELOCIDAD=30  → 1 min real = 30 min simulados.
VELOCIDAD_SIMULACION = float(os.getenv("KIRBOT_VELOCIDAD", "1"))

# ── Datos del negocio (ficticios: reemplázalos por los reales) ───────────────
NEGOCIO = {
    "nombre": "KIRBOT",
    "horario": "Lunes a domingo (incluidos festivos) de 12:00 m. a 10:00 p.m.",
    "direccion": "Calle Principal #45-20, Bucaramanga",
    "telefono": "+57 300 1234567",
    "whatsapp": "+57 300 1234567",
    "pagos": "efectivo, transferencias Nequi/Daviplata y tarjetas débito/crédito",
    "envio_costo": 4000,
    "envio_gratis_desde": 50000,
    "tiempo_domicilio_min": 35,
    "tiempo_llevar_min": 20,
    "tiempo_local_min": 15,
    "recargo_pizza_familiar": 12000,
}
