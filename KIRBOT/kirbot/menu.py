"""Menú de demostración y búsqueda de productos en texto libre."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .config import NEGOCIO


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni ñ, espacios simples."""
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).strip()


def pesos(valor: float) -> str:
    return "$" + f"{valor:,.0f}".replace(",", ".")


@dataclass(frozen=True)
class Producto:
    id: str
    nombre: str
    categoria: str
    precio: int
    descripcion: str
    ingredientes: tuple[str, ...]
    tags: tuple[str, ...] = ()
    alias: tuple[str, ...] = ()
    disponible: bool = True


CATEGORIAS = {
    "hamburguesas": "🍔 Hamburguesas",
    "pizzas": "🍕 Pizzas",
    "comidas_rapidas": "🌭 Comidas rápidas y ligeras",
    "bebidas": "🥤 Bebidas",
    "postres": "🍰 Postres",
    "adicionales": "➕ Adicionales",
}

PRODUCTOS: list[Producto] = [
    # Hamburguesas
    Producto("ham_clasica", "Hamburguesa Clásica", "hamburguesas", 18000,
             "Carne 100% de res, queso cheddar, lechuga, tomate y salsa especial.",
             ("carne de res", "queso cheddar", "lechuga", "tomate", "salsa de la casa"),
             ("jugosa", "abundante"), ("hamburguesa clasica", "clasica")),
    Producto("ham_doble", "Hamburguesa Doble Tocino", "hamburguesas", 24000,
             "Doble carne, doble tocino crujiente, queso suizo y cebolla caramelizada.",
             ("doble carne", "tocino", "queso suizo", "cebolla caramelizada"),
             ("estrella", "jugosa", "abundante", "tocino"),
             ("hamburguesa doble tocino", "hamburguesa doble", "doble tocino")),
    Producto("ham_crispy", "Hamburguesa Pollo Crispy", "hamburguesas", 19000,
             "Pechuga de pollo apanada crujiente, ensalada coleslaw y mayonesa de ajo.",
             ("pollo apanado", "coleslaw", "mayonesa de ajo"),
             ("pollo",), ("hamburguesa pollo crispy", "hamburguesa de pollo", "pollo crispy", "crispy")),
    Producto("ham_veggie", "Hamburguesa Veggie", "hamburguesas", 20000,
             "Medallón de lentejas y quinoa, queso mozzarella, rúcula y tomate.",
             ("lentejas", "quinoa", "queso mozzarella", "rúcula", "tomate"),
             ("vegetariano", "ligero"), ("hamburguesa veggie", "hamburguesa vegetariana", "veggie")),
    # Pizzas
    Producto("piz_napolitana", "Pizza Napolitana", "pizzas", 28000,
             "Base de tomate natural, mozzarella fresca, albahaca y aceite de oliva.",
             ("tomate", "mozzarella", "albahaca", "aceite de oliva"),
             ("vegetariano", "ninos", "compartir"), ("pizza napolitana", "napolitana")),
    Producto("piz_hawaiana", "Pizza Hawaiana", "pizzas", 30000,
             "Salsa de tomate, queso mozzarella, jamón ahumado y piña en almíbar.",
             ("mozzarella", "jamón", "piña"),
             ("ninos", "compartir"), ("pizza hawaiana", "hawaiana")),
    Producto("piz_pepperoni", "Pizza Pepperoni Suprema", "pizzas", 32000,
             "Doble porción de pepperoni artesanal y queso mozzarella fundido.",
             ("pepperoni", "mozzarella", "salsa de tomate"),
             ("estrella", "picante", "compartir", "intenso"),
             ("pizza pepperoni suprema", "pizza pepperoni", "pepperoni")),
    # Comidas rápidas y ligeras
    Producto("perro_especial", "Perro Caliente Especial", "comidas_rapidas", 14000,
             "Salchicha americana, queso fundido, papas ripio y chuleta de tocino.",
             ("salchicha", "queso", "papas ripio", "tocino"),
             ("rapido", "abundante", "tocino"), ("perro caliente especial", "perro caliente", "hot dog", "perro")),
    Producto("salchipapas", "Salchipapas de la Casa", "comidas_rapidas", 16000,
             "Papas a la francesa doradas, salchicha premium, queso costeño rallado y salsas.",
             ("papas fritas", "salchicha", "queso rallado", "salsa rosada, de ajo y de piña"),
             ("estrella", "rapido", "compartir", "abundante"), ("salchipapa de la casa", "salchipapa")),
    Producto("papas_fritas", "Papas Fritas Grandes", "comidas_rapidas", 8000,
             "Porción grande de papas a la francesa con sal marina.",
             ("papa", "sal marina"), ("rapido", "compartir", "vegetariano"),
             ("papas fritas grandes", "papa frita", "papa a la francesa", "papa")),
    Producto("sandwich_pollo", "Sándwich de Pollo", "comidas_rapidas", 17000,
             "Pollo desmechado, queso, lechuga y tomate en pan artesanal. Incluye papas.",
             ("pollo desmechado", "queso", "lechuga", "tomate", "pan artesanal"),
             ("pollo", "rapido"), ("sandwich de pollo", "sandwich")),
    Producto("menu_infantil", "Menú Infantil", "comidas_rapidas", 15000,
             "Mini hamburguesa, papas fritas y jugo de fruta.",
             ("mini hamburguesa", "papas fritas", "jugo de fruta"),
             ("ninos", "rapido"), ("menu infantil", "combo infantil", "menu para ninos")),
    Producto("ensalada", "Ensalada Fresca", "comidas_rapidas", 15000,
             "Lechugas mixtas, tomate cherry, pepino, zanahoria y aderezo de limón.",
             ("lechugas", "tomate cherry", "pepino", "zanahoria", "limón"),
             ("ligero", "vegetariano", "sin_gluten"), ("ensalada fresca", "ensalada")),
    # Bebidas
    Producto("coca", "Gaseosa Coca-Cola 400ml", "bebidas", 5000, "Bebida gaseosa refrescante.",
             ("agua carbonatada", "azúcar", "sabor cola"), ("refrescante",),
             ("coca cola", "coca", "gaseosa")),
    Producto("coca_zero", "Gaseosa Coca-Cola Zero 400ml", "bebidas", 5000, "Sabor cola sin azúcar.",
             ("agua carbonatada", "edulcorantes", "sabor cola"), ("refrescante", "ligero"),
             ("gaseosa zero", "coca cola zero", "coca zero", "zero")),
    Producto("malt_choc", "Malteada de Chocolate", "bebidas", 9000,
             "Helado artesanal de chocolate batido con leche fría y crema chantilly.",
             ("helado de chocolate", "leche", "crema chantilly"), ("estrella", "dulce"),
             ("malteada de chocolate",)),
    Producto("malt_van", "Malteada de Vainilla", "bebidas", 9000,
             "Helado artesanal de vainilla batido con leche fría y crema chantilly.",
             ("helado de vainilla", "leche", "crema chantilly"), ("dulce",),
             ("malteada de vainilla",)),
    Producto("limonada", "Limonada de Hierbabuena", "bebidas", 7000,
             "Limonada natural frappé con hojas frescas de hierbabuena.",
             ("limón", "hierbabuena", "azúcar", "hielo"), ("refrescante", "ligero"),
             ("limonada de hierbabuena", "limonada")),
    # Postres
    Producto("tres_leches", "Postre Tres Leches", "postres", 8500,
             "Bizcocho esponjoso bañado en tres tipos de leche con toque de canela.",
             ("leche condensada", "crema de leche", "leche entera", "bizcocho"), ("dulce",),
             ("postre tres leches", "tres leches")),
    Producto("brownie", "Brownie con Helado", "postres", 9500,
             "Brownie de chocolate tibio acompañado de una bola de helado de vainilla.",
             ("chocolate", "nueces", "helado de vainilla"), ("estrella", "dulce"),
             ("brownie con helado", "postre de chocolate", "brownie")),
    # Adicionales
    Producto("tocino_extra", "Tocino Extra", "adicionales", 3000, "Porción adicional de tocino crujiente.",
             ("tocino",), (), ("tocino extra", "tocino adicional")),
    Producto("queso_extra", "Queso Extra", "adicionales", 2500, "Porción adicional de queso.",
             ("queso",), (), ("queso extra", "queso adicional")),
]

POR_ID = {p.id: p for p in PRODUCTOS}
# normalizada → palabra original (recupera tildes y ñ al mostrar notas: "pina" → "piña")
PALABRAS = {normalizar(w): w for p in PRODUCTOS for ing in p.ingredientes for w in ing.split()}

# Alias genéricos que pueden referirse a varios productos (se pide aclaración).
ALIAS_GENERICOS = {
    "hamburguesa": ["ham_clasica", "ham_doble", "ham_crispy", "ham_veggie"],
    "pizza": ["piz_napolitana", "piz_hawaiana", "piz_pepperoni"],
    "malteada": ["malt_choc", "malt_van"],
    "postre": ["tres_leches", "brownie"],
}


def _patron(alias: str) -> re.Pattern:
    """Regex del alias aceptando plural en cada palabra (pizzas hawaianas)."""
    palabras = [re.escape(w) + (r"(?:s|es)?" if len(w) > 3 else "") for w in alias.split()]
    return re.compile(r"\b" + r" ".join(palabras) + r"\b")


def _construir_indice() -> list[tuple[str, re.Pattern, list[str]]]:
    indice: list[tuple[str, re.Pattern, list[str]]] = []
    for p in PRODUCTOS:
        for a in p.alias:
            n = normalizar(a)
            indice.append((n, _patron(n), [p.id]))
    for a, ids in ALIAS_GENERICOS.items():
        indice.append((a, _patron(a), ids))
    return sorted(indice, key=lambda x: -len(x[0]))  # más largo primero


_INDICE = _construir_indice()

NUMEROS = {"un": 1, "uno": 1, "una": 1, "unos": 1, "unas": 1, "otra": 1, "otro": 1, "dos": 2, "tres": 3,
           "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


@dataclass
class Mencion:
    productos: list[Producto]
    cantidad: int
    inicio: int

    @property
    def ambigua(self) -> bool:
        return len(self.productos) > 1


def _cantidad_previa(t: str, inicio: int) -> int:
    previos = t[:inicio].split()
    if previos:
        ult = previos[-1]
        if ult.isdigit():
            return max(1, min(int(ult), 20))
        if ult in NUMEROS:
            return NUMEROS[ult]
    return 1


def buscar_menciones(texto: str) -> list[Mencion]:
    """Encuentra productos nombrados en el texto, con su cantidad."""
    t = normalizar(texto)
    usado = [False] * len(t)
    menciones: list[Mencion] = []
    for _alias, patron, ids in _INDICE:
        for m in patron.finditer(t):
            if any(usado[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                usado[i] = True
            menciones.append(Mencion([POR_ID[i] for i in ids], _cantidad_previa(t, m.start()), m.start()))
    return sorted(menciones, key=lambda m: m.inicio)


def productos_de(categoria: str) -> list[Producto]:
    return [p for p in PRODUCTOS if p.categoria == categoria]


def precio_con_tamano(p: Producto, tamano: str | None) -> int:
    if p.categoria == "pizzas" and tamano == "familiar":
        return p.precio + NEGOCIO["recargo_pizza_familiar"]
    return p.precio


def a_json() -> dict:
    """Menú en formato dict (para guardarlo junto al modelo)."""
    out: dict[str, list[dict]] = {}
    for p in PRODUCTOS:
        out.setdefault(p.categoria, []).append(
            {"nombre": p.nombre, "descripcion": p.descripcion, "ingredientes": list(p.ingredientes),
             "precio": p.precio, "categoria": p.categoria, "disponible": p.disponible})
    return out
