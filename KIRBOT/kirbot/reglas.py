"""Capa de reglas que refuerza al modelo cuando el texto trae señales inequívocas."""
from __future__ import annotations

import re

from .menu import normalizar

# (patrón sobre texto normalizado sin tildes, peso)
REGLAS: dict[str, list[tuple[str, int]]] = {
    "consulta_pedidos": [
        (r"\b(domiciliario|rastre\w+)\b", 2),
        (r"\b(estado|en que va|por donde va|donde esta) (de )?(mi|el)? ?(pedido|orden|comida|domicilio|entrega)", 2),
        (r"\bno ha llegado|todavia no (ha )?llega|cuanto falta|cuanto (se )?(demora|tarda)|se (esta )?(demora|tarda)", 2),
        (r"\b(mi|el) (pedido|orden|domicilio|comida) (esta|va|sigue|se)\b", 1),
        (r"\b(demora|retraso|tarde|tardando|esperando|en camino|en cocina|numero de orden)\b", 1),
        (r"\bpor que (tarda|no ha llegado|no llega)", 2),
    ],
    "atencion_cliente": [
        (r"\b(horario|abren|cierran|abierto|cerrado)\b|a que hora (abren|cierran|atienden)", 2),
        (r"\b(donde (queda|estan|se ubican)|ubicacion|ubicados|direccion del)\b", 2),
        (r"\b(telefono|whatsapp|contacto|correo)\b", 2),
        (r"\b(metodos? de pago|aceptan|reciben|nequi|daviplata|tarjeta|transferencia)\b", 1),
        (r"\b(queja|reclam\w+|mal servicio|sugerencia|felicit\w+|excelente servicio|comentario)\b", 2),
        (r"\b(reserva\w*|cumpleanos|eventos?)\b", 2),
        (r"\b(parqueadero|parqueo|estacionamiento|pet friendly|mascotas?|factura\w*)\b", 2),
        (r"\b(promocion\w*|descuentos?|ofertas?|cupon\w*|reembolso|garantia|devolucion)\b", 2),
        (r"\b(asesor|humano|persona real)\b", 2),
        (r"\b(hacen|tienen|cobran|cuanto cuesta|cual es el costo del) (el )?(servicio a )?domicilios?\b", 2),
    ],
    "consulta_menu": [
        (r"\b(menu|carta)\b", 2),
        (r"\b(ingredientes?|que (lleva|trae|incluye|contiene|tiene)|descripcion)\b", 2),
        (r"\b(cuanto|cual) (cuesta|cuestan|vale|valen|es el (precio|valor))|\b(precios?|valor exacto)\b", 2),
        (r"\b(tamanos?|disponibilidad|disponibles?|sabores)\b", 1),
        (r"\btienen\b", 1),
    ],
    "gestion_pedidos": [
        (r"\bquiero (pedir|ordenar|hacer un pedido|finalizar|cancelar|revisar|modificar|cambiar)", 2),
        (r"\b(quisiera|deseo) (ordenar|pedir|confirmar)", 2),
        (r"\b(pon|ponga|deme|traeme)\b (una?|dos|tres|unas?|unos?|\d)", 2),
        (r"\b(agrega|agregame|anade|anademe|ponme|pongame|sacame|sacale|quita|quitame|quitale|elimina|borra)\b", 2),
        (r"\b(confirm\w+|finaliz\w+|proceder|eso seria todo|es todo|carrito|cuanto llevo|total de lo que llevo)\b", 2),
        (r"\b(para llevar|a domicilio|pago (con|en)|con pago|en efectivo)\b", 1),
        (r"\b(mi orden|mi pedido) (antes|actual)\b", 1),
        (r"\bquiero (una?|dos|tres|unas?) \w+", 1),
    ],
    "recomendacion_comidas": [
        (r"\b(recomiend\w*|recomienda\w*|sugier\w*|sugerenc\w*|aconsej\w*)\b", 2),
        (r"\b(no se que (pedir|comer)|ayudame a (escoger|elegir|decidir|armar)|que pido|que le pido)\b", 2),
        (r"\b(especialidad|mas vendid\w+|plato estrella|plato fuerte|plato especial|que esta rico|lo mas rico)\b", 2),
        (r"\b(antojo|opciones (vegetarian\w+|sin gluten|saludables?))\b", 1),
    ],
    "fuera_de_tema": [
        (r"\b(capital de|futbol|partido|mundial|pelicula|serie|libro|novela|cancion|chiste|python|programa\w*)\b", 2),
        (r"\b(clima|temperatura|planetas?|dinosaurios?|gravedad|luna|sol|sistema solar|matematicas?|ecuacion)\b", 2),
        (r"\b(iphone|celular|dolar|traduce\w*|presidente|tarea|entrevista de trabajo|inteligencia artificial)\b", 2),
        (r"\b(receta|como (preparar|cocinar|hacer|se hace|se programa) (una?|el|la|un)? ?\w+)\b", 2),
        (r"\b(pagina web|aeropuerto|historia de|bajo de peso|que hora es en|dato curioso)\b", 2),
    ],
}

_COMPILADAS = {k: [(re.compile(p), w) for p, w in v] for k, v in REGLAS.items()}


def puntuar(texto: str) -> dict[str, int]:
    t = normalizar(texto)
    return {cat: sum(w for rx, w in reglas if rx.search(t)) for cat, reglas in _COMPILADAS.items()}


def intencion_por_reglas(texto: str) -> tuple[str | None, int]:
    """Devuelve (categoría, puntaje) si hay una ganadora clara; si no, (None, 0)."""
    p = sorted(puntuar(texto).items(), key=lambda x: -x[1])
    (c1, s1), (_c2, s2) = p[0], p[1]
    if s1 >= 2 and s1 > s2:
        return c1, s1
    return None, 0
