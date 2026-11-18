"""Pruebas automáticas. Ejecuta:  pytest -q   (no necesitan GPU ni descargar modelos)."""
from datetime import datetime, timedelta

import pytest

from kirbot.clasificador import ClasificadorTfidf
from kirbot.datos import cargar_dataset, dividir
from kirbot.dialogo import Kirbot, Sesion
from kirbot.menu import buscar_menciones, normalizar


@pytest.fixture(scope="module")
def clf():
    df = cargar_dataset()
    return ClasificadorTfidf.entrenar(df["texto"], list(df["categoria"]))


@pytest.fixture
def reloj():
    return {"ahora": datetime(2026, 1, 1, 12, 0)}


@pytest.fixture
def kb(clf, reloj):
    return Kirbot(clf, ahora=lambda: reloj["ahora"])


def chat(kb, s, *mensajes):
    return [kb.responder(m, s) for m in mensajes]


# ── Datos ────────────────────────────────────────────────
def test_dataset_tiene_6_categorias_con_minimo_20():
    df = cargar_dataset()
    conteo = df["categoria"].value_counts()
    assert len(conteo) == 6 and conteo.min() >= 20


def test_division_estratificada_sin_fuga():
    df = cargar_dataset()
    tr, va, te, etiquetas = dividir(df)
    assert not (set(tr["texto"]) & set(te["texto"])) and not (set(tr["texto"]) & set(va["texto"]))
    assert len(etiquetas) == 6


def test_clasificador_tfidf_supera_el_azar_por_mucho(clf):
    df = cargar_dataset()
    _, _, te, _ = dividir(df)
    aciertos = sum(p.categoria == c for p, c in zip(clf.predecir_lote(list(te["texto"])), te["categoria"]))
    assert aciertos / len(te) > 0.55


# ── Menú ─────────────────────────────────────────────────
def test_menciones_con_cantidad_y_plural():
    m = buscar_menciones("quiero 3 pizzas hawaianas y dos gaseosas zero")
    assert [(x.productos[0].id, x.cantidad) for x in m] == [("piz_hawaiana", 3), ("coca_zero", 2)]


def test_generico_es_ambiguo():
    (m,) = buscar_menciones("una hamburguesa")
    assert m.ambigua and len(m.productos) == 4


def test_normalizar():
    assert normalizar("  ¿Piña,  Ñandú? ") == "¿pina, nandu?"


# ── Diálogo ──────────────────────────────────────────────
def test_saludo_y_agradecimiento(kb):
    s = Sesion()
    r1, r2 = chat(kb, s, "hola", "muchas gracias")
    assert "KIRBOT" in r1.texto and "gusto" in r2.texto


def test_recomendacion_respeta_filtros(kb):
    r = kb.responder("Busco una hamburguesa jugosa sin cebolla", Sesion())
    assert "Clásica" in r.texto and "Doble Tocino" not in r.texto.split("Combínalo")[0]
    r = kb.responder("Tienes alguna recomendación vegetariana", Sesion())
    assert "Veggie" in r.texto or "Napolitana" in r.texto


def test_precio_e_ingredientes(kb):
    assert "$28.000" in kb.responder("Cuánto cuesta la pizza napolitana", Sesion()).texto
    assert "carne de res" in kb.responder("Qué ingredientes tiene la hamburguesa clásica", Sesion()).texto


def test_flujo_completo_de_pedido_y_estado(kb, reloj):
    s = Sesion()
    chat(kb, s, "Quisiera ordenar una hamburguesa clásica sin cebolla y dos gaseosas")
    assert [(l.producto_id, l.cantidad, l.notas) for l in s.carrito] == \
           [("ham_clasica", 1, ["sin cebolla"]), ("coca", 2, [])]
    assert s.subtotal == 28000
    chat(kb, s, "confirmar mi pedido")
    assert s.esperando is None and s.modalidad is None                # pregunta modalidad
    chat(kb, s, "a domicilio")
    assert s.esperando == "direccion"
    chat(kb, s, "Calle 45 #20-30 Cabecera")
    assert s.direccion
    chat(kb, s, "efectivo")
    assert s.esperando == "confirmacion"
    r = kb.responder("sí", s)
    assert "#101" in r.texto and not s.carrito and len(s.pedidos) == 1
    assert s.pedidos[0].envio == 4000 and s.pedidos[0].total == 32000   # < $50.000 paga domicilio

    reloj["ahora"] += timedelta(minutes=5)
    assert "En preparación" in kb.responder("¿cómo va mi pedido?", s).texto
    reloj["ahora"] += timedelta(minutes=60)
    assert "Entregado" in kb.responder("estado de mi orden 101", s).texto
    assert "No encuentro" in kb.responder("estado de mi orden 999", s).texto


def test_ambiguedad_se_resuelve_en_el_siguiente_mensaje(kb):
    s = Sesion()
    r = kb.responder("quiero una malteada", s)
    assert "Cuál" in r.texto and s.pendiente
    kb.responder("de vainilla", s)
    assert [l.producto_id for l in s.carrito] == ["malt_van"]


def test_quitar_cambiar_y_cancelar(kb):
    s = Sesion()
    chat(kb, s, "quiero dos hamburguesas clásicas y una limonada")
    chat(kb, s, "cambia la limonada por una gaseosa zero")
    assert {l.producto_id for l in s.carrito} == {"ham_clasica", "coca_zero"}
    chat(kb, s, "cancelar el último producto")
    assert [l.producto_id for l in s.carrito] == ["ham_clasica"]
    chat(kb, s, "quiero cancelar mi pedido")
    assert not s.carrito


def test_pedido_vacio_no_se_confirma(kb):
    assert "vacío" in kb.responder("quiero confirmar mi pedido", Sesion()).texto


def test_fuera_de_tema_redirige(kb):
    r = kb.responder("Cuál es la capital de Japón", Sesion())
    assert r.categoria == "fuera_de_tema" and "menú" in r.texto


def test_queja_queda_registrada(kb):
    s = Sesion()
    kb.responder("Quiero poner una queja sobre mi servicio", s)
    assert len(s.quejas) == 1


def test_mensaje_vacio_y_sin_sentido_no_rompen(kb):
    assert kb.responder("   ", Sesion()).categoria == "vacio"
    assert kb.responder("asdfgh qwerty", Sesion()).texto
