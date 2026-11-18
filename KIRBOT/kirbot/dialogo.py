"""Lógica conversacional de KIRBOT: intención → respuesta, con memoria de sesión (carrito, pedido, contexto)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .clasificador import Clasificador, Prediccion
from .config import NEGOCIO, VELOCIDAD_SIMULACION
from .menu import (CATEGORIAS, NUMEROS, PALABRAS, POR_ID, PRODUCTOS, Mencion, Producto, buscar_menciones, normalizar, pesos,
                   precio_con_tamano, productos_de)
from .reglas import intencion_por_reglas

PRINCIPALES = ["hamburguesas", "pizzas", "comidas_rapidas"]
MODALIDADES = {"domicilio": "a domicilio 🛵", "llevar": "para llevar 🛍️", "local": "para comer en el local 🍽️"}

BIENVENIDA = (
    "🍔 ¡Hola! Soy **KIRBOT**, tu asistente del restaurante. Puedo:\n\n"
    "- 🍽️ **Recomendarte** platos según tus gustos\n"
    "- 📜 Mostrarte el **menú**, precios e ingredientes\n"
    "- 🛒 **Tomar tu pedido** (domicilio, para llevar o en el local)\n"
    "- 📦 Contarte **cómo va tu pedido**\n"
    "- ℹ️ Resolver dudas de horarios, pagos, ubicación y más\n\n"
    "¿Por dónde empezamos?"
)


# ══════════════════════ Estructuras de datos ══════════════════════
@dataclass
class Linea:
    producto_id: str
    cantidad: int = 1
    tamano: str | None = None
    notas: list[str] = field(default_factory=list)

    @property
    def producto(self) -> Producto:
        return POR_ID[self.producto_id]

    @property
    def nombre(self) -> str:
        return f"{self.producto.nombre} ({self.tamano})" if self.tamano else self.producto.nombre

    @property
    def subtotal(self) -> int:
        return precio_con_tamano(self.producto, self.tamano) * self.cantidad


@dataclass
class Pedido:
    numero: int
    lineas: list[Linea]
    modalidad: str
    pago: str
    direccion: str | None
    creado: datetime
    subtotal: int
    envio: int

    @property
    def total(self) -> int:
        return self.subtotal + self.envio


@dataclass
class Sesion:
    carrito: list[Linea] = field(default_factory=list)
    modalidad: str | None = None
    pago: str | None = None
    direccion: str | None = None
    esperando: str | None = None          # None | 'direccion' | 'confirmacion'
    pedidos: list[Pedido] = field(default_factory=list)
    quejas: list[str] = field(default_factory=list)
    ultimos: list[str] = field(default_factory=list)   # últimos productos mostrados (para "agrégala")
    pendiente: dict | None = None                        # pregunta de aclaración abierta {ids, cantidad, notas}

    @property
    def subtotal(self) -> int:
        return sum(l.subtotal for l in self.carrito)

    def reiniciar(self):
        self.__init__()


@dataclass
class Respuesta:
    texto: str
    categoria: str
    confianza: float = 1.0
    fuente: str = "reglas"
    probabilidades: dict[str, float] = field(default_factory=dict)


# ══════════════════════ Utilidades de texto ══════════════════════
RX_AFIRMA = re.compile(r"^(si|sii+|dale|claro|ok|okay|de una|listo|perfecto|va|vale|bueno|correcto|confirmo|"
                       r"confirmar|confirma|por supuesto|eso es|exacto)\b")
RX_NIEGA = re.compile(r"^(no|nop|nel|mejor no|cancel\w*|espera|aun no|todavia no)\b")
RX_QUITAR = re.compile(r"\b(quita|quitar|quitame|quitale|quite|elimina|eliminar|saca|sacar|sacame|sacale|borra|borrar|retira|"
                       r"retirar|cancela|cancelar|cancelo)\b")
RX_CONFIRMAR = re.compile(r"\b(confirm\w+|finaliz\w+|proceder|procede|pagar|eso seria todo|es todo|ya no mas|"
                          r"enviar mi pedido|cerrar (mi )?(pedido|orden)|listo)\b")
RX_DIRECCION = re.compile(r"((?:calle|carrera|cra|cl|kr|avenida|av|diagonal|transversal)\.?\s*#?\s*\d+.*)", re.I)


def _lista(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def _notas_sin(t: str) -> list[str]:
    """«sin cebolla ni tomate» → ['sin cebolla', 'sin tomate'] (ignora cantidades y productos)."""
    notas = []
    for m in re.finditer(r"\bsin ((?:\w+)(?:(?:,| ni| y) \w+)*)", t):
        for i, x in enumerate(re.split(r",| ni | y ", m.group(1))):
            x = x.strip()
            if not x or x in NUMEROS or x in ("mas", "problema", "falta", "pedir", "con", "para", "otra", "otro"):
                continue
            if buscar_menciones(x) or (i > 0 and x not in PALABRAS):   # los siguientes deben ser ingredientes
                continue
            notas.append(f"sin {PALABRAS.get(x, x)}")
    return notas


def _pago_txt(p: str) -> str:
    return "pago en efectivo" if p == "efectivo" else f"pago con {p}"


def _hay_slot(texto: str, t: str) -> bool:
    """¿El mensaje solo aporta datos de entrega/pago? (domicilio, efectivo, una dirección…)"""
    return bool(RX_DIRECCION.search(texto) or re.search(
        r"\b(domicilio|llevar|recoger|recojo|en el local|comer aqui|efectivo|nequi|daviplata|tarjeta|credito|"
        r"debito|transferencia)\b", t))


# ══════════════════════ Motor de diálogo ══════════════════════
class Kirbot:
    def __init__(self, clasificador: Clasificador, ahora: Callable[[], datetime] = datetime.now):
        self.clf = clasificador
        self.ahora = ahora
        self._numero = 100

    # ── Punto de entrada ───────────────────────────────────────
    def responder(self, texto: str, s: Sesion) -> Respuesta:
        texto = (texto or "").strip()
        if not texto:
            return Respuesta("Escríbeme tu mensaje y te ayudo 😊", "vacio")
        t = normalizar(texto)

        if s.esperando:                                  # flujo de checkout en curso
            r = self._continuar_flujo(texto, t, s)
            if r:
                return Respuesta(r, "gestion_pedidos", 1.0, "flujo")

        if s.pendiente:
            r = self._resolver_pendiente(t, s)
            if r:
                return Respuesta(r, "gestion_pedidos", 1.0, "contexto")

        r = self._charla(t, s)
        if r:
            return Respuesta(r, "charla", 1.0, "reglas")

        if s.carrito and len(t.split()) <= 6 and _hay_slot(texto, t) and not buscar_menciones(texto):
            return Respuesta(self._gestion(texto, t, s), "gestion_pedidos", 0.9, "contexto")

        pred = self._clasificar(texto)
        cat = pred.categoria

        if pred.fuente == "modelo" and pred.confianza < self.clf.umbral_confianza:
            return Respuesta(self._aclarar(texto, pred, s), "sin_claridad", pred.confianza, pred.fuente,
                             pred.probabilidades)

        manejadores = {
            "recomendacion_comidas": self._recomendar,
            "consulta_menu": self._menu,
            "gestion_pedidos": self._gestion,
            "consulta_pedidos": self._pedidos,
            "atencion_cliente": self._atencion,
            "fuera_de_tema": self._fuera,
        }
        texto_resp = manejadores.get(cat, self._fuera)(texto, t, s)
        return Respuesta(texto_resp, cat, pred.confianza, pred.fuente, pred.probabilidades)

    # ── Clasificación híbrida (modelo + reglas) ────────────────
    def _clasificar(self, texto: str) -> Prediccion:
        pred = self.clf.predecir(texto)
        regla, puntaje = intencion_por_reglas(texto)
        if regla is None:
            return pred
        if regla == pred.categoria:
            return Prediccion(regla, max(pred.confianza, 0.75), pred.probabilidades, "modelo+reglas")
        if pred.confianza < 0.85 or puntaje >= 3:
            return Prediccion(regla, min(0.9, 0.6 + 0.1 * puntaje), pred.probabilidades, "reglas")
        return pred

    # ── Charla básica ──────────────────────────────────────────
    def _charla(self, t: str, s: Sesion) -> str | None:
        if re.fullmatch(r"(hola|holi|ola|hey|buenas|buenos dias|buen dia|buenas tardes|buenas noches|saludos|"
                        r"que mas|quiubo)( \w+){0,2}", t):
            return BIENVENIDA
        if re.search(r"\b(gracias|muy amable|te agradezco|mil gracias)\b", t) and len(t.split()) <= 6:
            return "¡Con mucho gusto! 😊 ¿Te ayudo con algo más?"
        if re.fullmatch(r"(chao|adios|hasta luego|nos vemos|bye|hasta pronto)\b.*", t):
            extra = " Tu pedido sigue abierto por si quieres retomarlo." if s.carrito else ""
            return f"¡Hasta pronto! 👋 Gracias por escribirle a KIRBOT.{extra}"
        if re.fullmatch(r"(ayuda|ayudame|necesito ayuda|quien eres|que eres|que puedes hacer|que haces|"
                        r"como funcionas|eres un bot|que sabes hacer)", t):
            return BIENVENIDA
        o = re.fullmatch(r"(?:la |el |lo )?(primer|segund|tercer|cuart|ultim)[ao]s?( opcion)?( por favor)?", t)
        if o and s.ultimos:
            i = {"primer": 0, "segund": 1, "tercer": 2, "cuart": 3, "ultim": -1}[o.group(1)]
            if -len(s.ultimos) <= i < len(s.ultimos):
                return self._agregar([(POR_ID[s.ultimos[i]], 1, [])], s, "")
        if s.ultimos and re.fullmatch(r"(si|sii+|dale|claro|ok|okay|de una|perfecto|va|vale|bueno)"
                                      r"( por favor| gracias)?", t):
            if len(s.ultimos) == 1:
                return self._agregar([(POR_ID[s.ultimos[0]], 1, [])], s, "")
            return "¿Cuál te gustaría? " + ", ".join(f"**{POR_ID[i].nombre}**" for i in s.ultimos) + \
                   ". Dime el nombre o «la primera», «la segunda»…"
        return None

    # ── Aclaración cuando el modelo no está seguro ─────────────
    def _aclarar(self, texto: str, pred: Prediccion, s: Sesion) -> str:
        menciones = [m for m in buscar_menciones(texto) if not m.ambigua]
        if menciones:
            p = menciones[0].productos[0]
            s.ultimos = [p.id]
            return (f"Veo que mencionas **{p.nombre}** ({pesos(p.precio)}). ¿Quieres **pedirla** o prefieres "
                    f"saber más (ingredientes, precio)?")
        frases = {
            "recomendacion_comidas": "que te **recomiende** algo de comer",
            "consulta_menu": "ver el **menú** o los precios",
            "gestion_pedidos": "**hacer o modificar un pedido**",
            "consulta_pedidos": "saber **cómo va tu pedido**",
            "atencion_cliente": "información del restaurante (horarios, pagos, ubicación…)",
            "fuera_de_tema": None,
        }
        top = [c for c, _ in sorted(pred.probabilidades.items(), key=lambda x: -x[1]) if frases.get(c)][:2]
        opciones = " o ".join(frases[c] for c in top) or "pedir, ver el menú o resolver dudas"
        return (f"🤔 No estoy seguro de haberte entendido. ¿Quieres {opciones}? "
                f"Cuéntamelo con otras palabras y te ayudo.")

    # ══════════════ RECOMENDACIONES ══════════════
    _TAGS = [
        (r"vegetarian|sin carne|no (come|como) carne|veggie|vegan", "vegetariano", "vegetariana"),
        (r"\bpollo\b", "pollo", "con pollo"),
        (r"picante|pican", "picante", "picante"),
        (r"\bnin[oa]s?\b|\bhij[oa]s?\b|pequen|infantil|\bbebe", "ninos", "para niños"),
        (r"ligero|liviano|saludable|gluten|grasos[oa]|dieta|light|\bfit\b", "ligero", "ligera"),
        (r"rapido|apuro|prisa|ya mismo", "rapido", "rápida"),
        (r"compartir|pareja|amigos|familia|dos personas", "compartir", "para compartir"),
        (r"hambre|llene|llenador|abundante|contundente", "abundante", "abundante"),
        (r"jugos[oa]", "jugosa", "jugosa"),
        (r"refresc|\bfresc[oa]", "refrescante", "refrescante"),
        (r"dulce|postre", "dulce", "dulce"),
        (r"tocino", "tocino", "con tocino"),
        (r"estrella|mas vendid|especialidad|favorit|lo mejor", "estrella", "la más pedida"),
    ]
    _CATS = [
        (r"hamburguesa", "hamburguesas"), (r"pizza", "pizzas"), (r"postre", "postres"),
        (r"bebida|tomar|beber|gaseosa|malteada|limonada|refresc", "bebidas"),
        (r"comida rapida|perro|papas|salchi|sandwich|ensalada", "comidas_rapidas"),
    ]

    def _recomendar(self, texto: str, t: str, s: Sesion) -> str:
        if re.search(r"combinacion|combo|armar", t):
            return self._combo(s)

        tags = [(tag, etq) for rx, tag, etq in self._TAGS if re.search(rx, t)]
        cats_pedidas = [c for rx, c in self._CATS if re.search(rx, t)]
        pairing = bool(re.search(r"combina|acompan|va bien con|maridaj", t))
        barato = bool(re.search(r"barat|economic|presupuesto|plata", t))
        excl = [x for n in _notas_sin(t) for x in [n.replace("sin ", "")]]

        if pairing and "bebidas" in cats_pedidas:
            cats = ["bebidas"]
        elif cats_pedidas:
            cats = cats_pedidas
        elif any(tg == "dulce" for tg, _ in tags):
            cats = ["postres", "bebidas"]
        elif any(tg == "refrescante" for tg, _ in tags):
            cats = ["bebidas"]
        else:
            cats = PRINCIPALES
        sin_filtros = not tags and not cats_pedidas and not barato

        cands = [p for p in PRODUCTOS if p.categoria in cats and p.disponible]
        cands = [p for p in cands if not any(normalizar(x) in normalizar(" ".join(p.ingredientes)) for x in excl)]
        req = [tg for tg, _ in tags]

        def score(p: Producto) -> int:
            return sum(tg in p.tags for tg in req)

        indice = {p.id: i for i, p in enumerate(PRODUCTOS)}
        parcial = False
        if sin_filtros:
            elegidos = [p for p in PRODUCTOS if "estrella" in p.tags][:4]
        else:
            estrictos = [p for p in cands if score(p) == len(req)] if req else list(cands)
            if estrictos:
                pool = estrictos
            else:
                pool = [p for p in cands if score(p) > 0]
                parcial = True
                if not pool:
                    pool = cands
            clave = (lambda p: (p.precio, indice[p.id])) if barato else \
                    (lambda p: (-score(p), "estrella" not in p.tags, indice[p.id]))
            pool = sorted(pool, key=clave)
            if len(cats_pedidas) >= 2 and not pairing:      # comparación: uno por categoría
                elegidos, vistos = [], set()
                for p in pool:
                    if p.categoria not in vistos:
                        vistos.add(p.categoria)
                        elegidos.append(p)
            else:
                elegidos = pool[:3]

        if not elegidos:
            return ("🤔 No encontré algo que cumpla exactamente eso en el menú de hoy. ¿Quieres que te muestre "
                    "el **menú completo** o te sugiera mis favoritos?")

        etiquetas = [e for _, e in tags] + [f"sin {x}" for x in excl]
        if sin_filtros:
            cabecera = "🍽️ **Mis favoritos de la casa:**"
        elif parcial:
            cabecera = ("🍽️ No tengo algo que cumpla *todo* eso, pero esto se acerca"
                        f"{' (' + ', '.join(etiquetas) + ')' if etiquetas else ''}:")
        else:
            cabecera = f"🍽️ Para lo que buscas{' (' + ', '.join(etiquetas) + ')' if etiquetas else ''}, te recomiendo:"
        if pairing:
            ref = [m.productos[0].nombre for m in buscar_menciones(texto) if not m.ambigua]
            cabecera = f"🥤 Para acompañar{' tu **' + ref[0] + '**' if ref else ''}, estas bebidas van muy bien:"

        lineas = [f"**{p.nombre}** ({pesos(p.precio)}) — {p.descripcion}" for p in elegidos]
        s.ultimos = [p.id for p in elegidos]
        out = cabecera + "\n\n" + _lista(lineas)

        if all(p.categoria in PRINCIPALES for p in elegidos):
            out += "\n\n💡 Combínalo con una **Limonada de Hierbabuena** ($7.000) o una **Malteada de Chocolate** ($9.000)."
        if re.search(r"gluten|alerg|celiac", t):
            out += "\n\n⚠️ Si tienes alergias o celiaquía, confírmalo con el personal: la cocina es compartida."
        if len(elegidos) == 1:
            out += f"\n\n¿Te la agrego a tu pedido? 🛒"
        else:
            out += "\n\n¿Cuál te provoca? Dime «agrega la primera» (o el nombre) y la sumo a tu pedido 🛒"
        return out

    def _combo(self, s: Sesion) -> str:
        plato, acomp, beb = POR_ID["ham_doble"], POR_ID["papas_fritas"], POR_ID["malt_choc"]
        total = plato.precio + acomp.precio + beb.precio
        s.ultimos = [plato.id, acomp.id, beb.id]
        return ("🍽️ **Combo perfecto de almuerzo** (armado por el chef):\n\n"
                f"- **{plato.nombre}** — {pesos(plato.precio)}\n"
                f"- **{acomp.nombre}** — {pesos(acomp.precio)}\n"
                f"- **{beb.nombre}** — {pesos(beb.precio)}\n\n"
                f"**Total: {pesos(total)}**. ¿Te lo agrego completo? Responde «sí» o dime qué cambiarías 🛒")

    # ══════════════ MENÚ ══════════════
    def _menu(self, texto: str, t: str, s: Sesion) -> str:
        if re.search(r"tipo de queso|que queso|queso usan|salsas?\b", t) and "salchipapa" not in t.split():
            if "salsa" in t:
                return ("🥫 Nuestras salsas de la casa son **rosada, de ajo y de piña**, y vienen incluidas con las "
                        "**Salchipapas**. Puedes pedir extra al hacer tu pedido.")
            return ("🧀 En las **pizzas** usamos queso **mozzarella** fresco; en las hamburguesas, **cheddar** "
                    "(Clásica) y **suizo** (Doble Tocino).")
        menciones = buscar_menciones(texto)
        especificas = [m for m in menciones if not m.ambigua]
        if especificas:
            menciones = especificas
        pide_precio = re.search(r"cuanto|precio|valor|cuesta|cuestan|vale|valen|costo", t)
        pide_ingr = re.search(r"ingredient|lleva|trae|contiene|incluye|viene con|que tiene|que son", t)
        pide_disp = re.search(r"\b(tienen|hay|disponib\w*)\b", t)
        pide_tam = re.search(r"tamano|familiar|mediana|porciones?", t)

        if pide_tam and not especificas and any(m.ambigua and m.productos[0].categoria == "pizzas" for m in menciones):
            return self._tamanos()
        if menciones:
            bloques, ids = [], []
            for m in menciones:
                if m.ambigua:
                    bloques.append("Estas son las opciones:\n" +
                                   _lista([f"**{p.nombre}** — {pesos(p.precio)}" for p in m.productos]))
                    ids += [p.id for p in m.productos]
                    continue
                p = m.productos[0]
                ids.append(p.id)
                if not p.disponible:
                    bloques.append(f"😕 **{p.nombre}** no está disponible por ahora.")
                elif pide_ingr and not pide_precio:
                    bloques.append(f"🧾 **{p.nombre}** lleva: {', '.join(p.ingredientes)}.\n"
                                   f"_Lo que no aparece en la lista no viene incluido; puedes pedirlo como adicional._")
                elif pide_precio and not pide_ingr:
                    extra = ""
                    if p.categoria == "pizzas":
                        extra = f" (mediana; familiar {pesos(p.precio + NEGOCIO['recargo_pizza_familiar'])})"
                    bloques.append(f"💲 **{p.nombre}**: {pesos(p.precio)}{extra}.")
                else:
                    pre = "✅ Sí, tenemos disponible: " if pide_disp else ""
                    bloques.append(f"{pre}**{p.nombre}** — {pesos(p.precio)}\n{p.descripcion}\n"
                                   f"_Ingredientes: {', '.join(p.ingredientes)}._")
            s.ultimos = ids[:4]
            if pide_tam and any(POR_ID[i].categoria == "pizzas" for i in ids):
                bloques.append(self._tamanos())
            return "\n\n".join(bloques) + "\n\n¿Quieres que la agregue a tu pedido? 🛒"

        if pide_tam or re.search(r"pizza familiar", t):
            return self._tamanos()
        cats = [c for rx, c in [(r"postre", "postres"), (r"bebida|gaseosa|tomar", "bebidas"),
                                (r"comida rapida|papas|perro|salchi", "comidas_rapidas"),
                                (r"adicional|extra", "adicionales")] if re.search(rx, t)]
        if cats:
            s.ultimos = [p.id for c in cats for p in productos_de(c)][:4]
            return "\n\n".join(self._seccion(c) for c in cats) + "\n\n¿Te ayudo a pedir algo? 🛒"
        return self._menu_completo()

    def _tamanos(self) -> str:
        return (f"📏 Las **pizzas** vienen en tamaño **mediana** (precio del menú) y **familiar** "
                f"(+{pesos(NEGOCIO['recargo_pizza_familiar'])}). Al pedir, dime «pizza hawaiana familiar».")

    def _seccion(self, cat: str) -> str:
        items = [f"**{p.nombre}** — {pesos(p.precio)} · _{p.descripcion}_" for p in productos_de(cat) if p.disponible]
        return f"**{CATEGORIAS[cat]}**\n" + _lista(items)

    def _menu_completo(self) -> str:
        secciones = "\n\n".join(self._seccion(c) for c in CATEGORIAS)
        return ("📜 **Nuestro menú**\n\n" + secciones +
                "\n\nPregúntame por un plato para ver sus ingredientes, o dime qué quieres pedir 🛒")

    # ══════════════ PEDIDOS: carrito y checkout ══════════════
    def _extraer_slots(self, texto: str, t: str, s: Sesion) -> dict[str, str]:
        cambios: dict[str, str] = {}
        m = RX_DIRECCION.search(texto)
        if m and len(m.group(1)) >= 6:
            s.direccion = m.group(1).strip()
            cambios["dirección"] = s.direccion
        mod = None
        if re.search(r"\b(domicilio|a mi casa|enviamelo|mandamelo|envio|a la puerta)\b", t) or m:
            mod = "domicilio"
        elif re.search(r"\b(llevar|recoger|recojo|recogerlo|paso por)\b", t):
            mod = "llevar"
        elif re.search(r"\b(en el local|comer aqui|en el restaurante|en mesa)\b", t):
            mod = "local"
        if mod and mod != s.modalidad:
            s.modalidad = mod
            cambios["modalidad"] = MODALIDADES[mod]
        pago = None
        if "efectivo" in t:
            pago = "efectivo"
        elif "nequi" in t:
            pago = "Nequi"
        elif "daviplata" in t:
            pago = "Daviplata"
        elif re.search(r"tarjeta|credito|debito|datafono", t):
            pago = "tarjeta"
        elif "transferencia" in t:
            pago = "transferencia"
        if pago and pago != s.pago:
            s.pago = pago
            cambios["pago"] = _pago_txt(pago)
        return cambios

    def _envio(self, s: Sesion | None = None, subtotal: int = 0, modalidad: str | None = None) -> int:
        modalidad = modalidad or (s.modalidad if s else None)
        subtotal = subtotal or (s.subtotal if s else 0)
        if modalidad != "domicilio" or subtotal >= NEGOCIO["envio_gratis_desde"]:
            return 0
        return NEGOCIO["envio_costo"]

    def _lineas_texto(self, lineas: list[Linea]) -> str:
        out = []
        for l in lineas:
            nota = f" _({', '.join(l.notas)})_" if l.notas else ""
            out.append(f"{l.cantidad} × {l.nombre}{nota} — {pesos(l.subtotal)}")
        return _lista(out)

    def _resumen(self, s: Sesion, detallado: bool = False) -> str:
        envio = self._envio(s)
        txt = "🛒 **Tu pedido actual:**\n" + self._lineas_texto(s.carrito) + f"\n\nSubtotal: **{pesos(s.subtotal)}**"
        if s.modalidad == "domicilio":
            txt += f"\nDomicilio: **{'gratis 🎉' if envio == 0 else pesos(envio)}**"
        if detallado or s.modalidad:
            txt += f"\n**Total: {pesos(s.subtotal + envio)}**"
        return txt

    @staticmethod
    def _cola(s: Sesion) -> str:
        return "\n\n¿Algo más o **confirmamos** tu pedido?" if s.carrito else ""

    def _agregar_lineas(self, s: Sesion, p: Producto, cantidad: int, tamano: str | None, notas: list[str]):
        for l in s.carrito:
            if l.producto_id == p.id and l.tamano == tamano and sorted(l.notas) == sorted(notas):
                l.cantidad += cantidad
                return
        s.carrito.append(Linea(p.id, cantidad, tamano, list(notas)))

    def _agregar(self, items: list[tuple[Producto, int, list[str]]], s: Sesion, t: str) -> str:
        tamano = "familiar" if "familiar" in t else None
        for p, c, notas in items:
            self._agregar_lineas(s, p, c, tamano if p.categoria == "pizzas" else None,
                                 notas if p.categoria not in ("bebidas", "postres") else [])
        s.ultimos = []
        filas = []
        for p, c, notas in items:
            fam = " (familiar)" if tamano and p.categoria == "pizzas" else ""
            nota = f" _({', '.join(notas)})_" if notas and p.categoria not in ("bebidas", "postres") else ""
            filas.append(f"{c} × {p.nombre}{fam}{nota}")
        txt = "✅ **¡Agregado a tu pedido!**\n" + _lista(filas) + "\n\n" + self._resumen(s)
        if not any(l.producto.categoria == "bebidas" for l in s.carrito):
            txt += "\n\n💡 ¿Le sumo una bebida? Una **Limonada de Hierbabuena** va perfecta."
        return txt + "\n\n¿Algo más o **confirmamos** tu pedido?"

    def _resolver(self, m: Mencion, s: Sesion) -> Producto | None:
        if not m.ambigua:
            return m.productos[0]
        en_carrito = [p for p in m.productos if any(l.producto_id == p.id for l in s.carrito)]
        return en_carrito[0] if len(en_carrito) == 1 else None

    def _resolver_pendiente(self, t: str, s: Sesion) -> str | None:
        pend, s.pendiente = s.pendiente, None
        cand = [POR_ID[i] for i in pend["ids"]]
        vacias = {"de", "con", "la", "el", "los", "las", "un", "una"}
        toks = [set(normalizar(p.nombre).split()) - vacias for p in cand]
        comunes = set.intersection(*toks)
        palabras = set(t.split())
        coinciden = [p for p, tk in zip(cand, toks) if (tk - comunes) & palabras]
        elegido = coinciden[0] if len(coinciden) == 1 else None
        if elegido is None:
            for clave, i in (("primer", 0), ("segund", 1), ("tercer", 2), ("cuart", 3), ("ultim", -1)):
                if clave in t and -len(cand) <= i < len(cand):
                    elegido = cand[i]
        if elegido is None:
            return None
        return self._agregar([(elegido, pend["cantidad"], pend["notas"])], s, t)

    def _pregunta_ambigua(self, m: Mencion) -> str:
        return (f"🤔 ¿Cuál prefieres? (dime el nombre o «la primera», «la segunda»…)\n" +
                _lista([f"**{p.nombre}** — {pesos(p.precio)}" for p in m.productos]))

    def _gestion(self, texto: str, t: str, s: Sesion) -> str:
        menciones = buscar_menciones(texto)
        if any(not m.ambigua and m.productos[0].categoria == "adicionales" for m in menciones):
            # "tocino extra a mi hamburguesa": el genérico es contexto, no un producto a pedir
            menciones = [m for m in menciones if not m.ambigua]
        cambios = self._extraer_slots(texto, t, s)

        # 1) Quitar / cancelar
        if RX_QUITAR.search(t):
            if menciones:
                quitados, faltan = [], []
                for m in menciones:
                    p = self._resolver(m, s)
                    if p is None:
                        return self._pregunta_ambigua(m) if not m.ambigua or not s.carrito else \
                            "No veo ese producto en tu pedido. " + self._resumen(s)
                    lin = next((l for l in s.carrito if l.producto_id == p.id), None)
                    (quitados if lin else faltan).append(p.nombre)
                    if lin:
                        s.carrito.remove(lin)
                partes = []
                if quitados:
                    partes.append("🗑️ Listo, quité: " + ", ".join(quitados) + ".")
                if faltan:
                    partes.append(f"No tenías {', '.join(faltan)} en tu pedido.")
                return " ".join(partes) + ("\n\n" + self._resumen(s) + self._cola(s) if s.carrito
                                           else "\n\nTu pedido quedó vacío.")
            if not s.carrito:
                return "Tu pedido está vacío, no hay nada que quitar 🙂 ¿Qué te gustaría pedir?"
            if re.search(r"\bultim[oa]\b", t):
                l = s.carrito.pop()
                return f"🗑️ Quité **{l.nombre}**.\n\n" + (self._resumen(s) + self._cola(s) if s.carrito
                                                          else "Tu pedido quedó vacío.")
            m = re.search(r"\b(?:quit\w*|saca\w*)\s+(?:la |el |los |las |toda |todo )?(\w+)", t)
            if m and m.group(1) not in ("pedido", "orden", "producto", "todo", "carrito"):
                palabra = m.group(1)
                nota = f"sin {PALABRAS.get(palabra, palabra)}"
                destino = [l for l in s.carrito
                           if palabra in normalizar(" ".join(l.producto.ingredientes + (l.producto.nombre,)))]
                aviso = ""
                if not destino:
                    destino, aviso = [s.carrito[-1]], f" (no veo «{palabra}» en la receta, pero lo dejo anotado)"
                for l in destino:
                    if nota not in l.notas:
                        l.notas.append(nota)
                return (f"📝 Anotado: **{nota}** en {', '.join(l.nombre for l in destino)}{aviso}.\n\n"
                        + self._resumen(s) + self._cola(s))
            if re.search(r"\b(pedido|orden|todo|carrito)\b", t):
                s.carrito.clear()
                s.modalidad = s.pago = s.direccion = None
                return "🗑️ Cancelé tu pedido en armado. Cuando quieras, empezamos uno nuevo 😊"

        # 2) Cambiar un producto por otro
        if re.search(r"\bcambi\w+", t) and " por " in f" {t} " and len(menciones) >= 2:
            a, b = self._resolver(menciones[0], s), self._resolver(menciones[1], s)
            lin = next((l for l in s.carrito if a and l.producto_id == a.id), None)
            if lin and b and not menciones[1].ambigua:
                s.carrito.remove(lin)
                self._agregar_lineas(s, b, lin.cantidad, None, [])
                return f"🔄 Cambié **{a.nombre}** por **{b.nombre}**.\n\n" + self._resumen(s) + self._cola(s)

        # 3) Cambiar cantidad
        if menciones and re.search(r"\b(cambi\w+|modific\w+|actualiz\w+|mejor)\b", t):
            m = menciones[0]
            p = self._resolver(m, s)
            lin = next((l for l in s.carrito if p and l.producto_id == p.id), None)
            if lin:
                lin.cantidad = m.cantidad
                return f"🔢 Listo: ahora tienes {lin.cantidad} × **{lin.nombre}**.\n\n" + self._resumen(s) + self._cola(s)

        # 4) Agregar productos (lo claro se agrega; lo ambiguo se pregunta)
        if menciones:
            items, ambigua = [], None
            for i, m in enumerate(menciones):
                fin = menciones[i + 1].inicio if i + 1 < len(menciones) else len(t)
                notas = _notas_sin(t[m.inicio:fin])
                if m.ambigua:
                    ambigua = ambigua or (m, notas)
                else:
                    items.append((m.productos[0], m.cantidad, notas))
            txt = self._agregar(items, s, t) if items else ""
            if cambios and txt:
                txt = f"Anotado: {', '.join(cambios.values())}.\n\n" + txt
            if ambigua:
                m, notas = ambigua
                s.pendiente = {"ids": [p.id for p in m.productos], "cantidad": m.cantidad, "notas": notas}
                pregunta = self._pregunta_ambigua(m)
                txt = (re.split(r"\n\n(?:💡|¿Algo más)", txt)[0] + "\n\n" if txt else "") + pregunta
            return txt

        # 5) Referencias al contexto ("agrégala", "la segunda")
        if s.ultimos and re.search(r"\b(esa|ese|esta|este|primer[ao]|segund[ao]|tercer[ao]|ultim[ao]|agregal\w+|"
                                   r"anadel\w+|pidel\w+|la misma|lo mismo)\b", t):
            orden = {"primer": 0, "segund": 1, "tercer": 2}
            idx = next((v for k, v in orden.items() if k in t), -1 if "ultim" in t else 0)
            try:
                p = POR_ID[s.ultimos[idx]]
            except IndexError:
                p = POR_ID[s.ultimos[0]]
            return self._agregar([(p, 1, _notas_sin(t))], s, t)

        # 6) Revisar total / resumen
        if re.search(r"\b(total|cuanto llevo|revis\w+|resumen|que llevo|carrito)\b", t) and not RX_CONFIRMAR.search(t):
            if not s.carrito:
                return "🛒 Tu pedido está vacío por ahora. Dime qué te gustaría pedir 😊"
            return self._resumen(s, detallado=True) + "\n\n¿Confirmamos o agregas algo más?"

        # 7) Confirmar / finalizar
        if RX_CONFIRMAR.search(t):
            return self._checkout(s, confirmado=True)

        # 8) Solo datos de entrega/pago
        if cambios:
            if s.carrito:
                return f"Anotado: {', '.join(cambios.values())}.\n\n" + self._checkout(s, confirmado=False)
            return f"Anotado: {', '.join(cambios.values())}. ¿Qué te gustaría pedir? 🍔"

        # 9) Pedido genérico
        if s.carrito:
            return self._resumen(s) + "\n\n¿Agregamos algo más o confirmamos?"
        return ("🛒 ¡Claro, empecemos tu pedido! Dime qué quieres, por ejemplo:\n"
                "«una hamburguesa clásica y dos gaseosas» o «una pizza hawaiana familiar».\n"
                "Si no sabes qué pedir, pídeme una recomendación 😉")

    def _continuar_flujo(self, texto: str, t: str, s: Sesion) -> str | None:
        if s.esperando == "direccion":
            if RX_NIEGA.match(t) and not RX_DIRECCION.search(texto):
                s.esperando = None
                s.modalidad = None
                return "Sin problema, cancelé el domicilio. ¿Prefieres **para llevar** o **comer en el local**?"
            cambios = self._extraer_slots(texto, t, s)
            if s.modalidad != "domicilio":
                s.esperando = None
                return self._checkout(s, confirmado=False)
            if "dirección" not in cambios and len(texto) >= 6 and re.search(r"\d", texto):
                s.direccion = texto.strip()
            if s.direccion:
                s.esperando = None
                return f"📍 Dirección guardada: **{s.direccion}**.\n\n" + self._checkout(s, confirmado=False)
            s.esperando = None
            return None
        if s.esperando == "confirmacion":
            if RX_NIEGA.match(t):
                s.esperando = None
                return "Sin problema, tu pedido sigue abierto. ¿Qué quieres cambiar o agregar?"
            if RX_AFIRMA.match(t):
                return self._checkout(s, confirmado=True)
            cambios = self._extraer_slots(texto, t, s)
            if cambios:
                return f"Anotado: {', '.join(cambios.values())}.\n\n" + self._checkout(s, confirmado=False)
            s.esperando = None
        return None

    def _checkout(self, s: Sesion, confirmado: bool) -> str:
        if not s.carrito:
            s.esperando = None
            return ("🛒 Tu pedido está vacío. Dime qué te gustaría pedir (ej.: «una hamburguesa clásica y una "
                    "gaseosa») o pídeme una recomendación.")
        if not s.modalidad:
            return (self._resumen(s) + "\n\n¿Cómo lo prefieres? **A domicilio** 🛵, **para llevar** 🛍️ "
                    "o **para comer en el local** 🍽️")
        if s.modalidad == "domicilio" and not s.direccion:
            s.esperando = "direccion"
            return "🛵 ¡Perfecto, a domicilio! ¿Cuál es tu **dirección de entrega** (calle/carrera, número y barrio)?"
        if not s.pago:
            return f"💳 ¿Cómo vas a pagar? Aceptamos {NEGOCIO['pagos']}."
        if not confirmado:
            s.esperando = "confirmacion"
            envio = self._envio(s)
            extra = f"\n📍 Dirección: {s.direccion}" if s.modalidad == "domicilio" else ""
            return (self._resumen(s, detallado=True) + f"\n\nModalidad: **{MODALIDADES[s.modalidad]}**{extra}"
                    f"\nPago: **{_pago_txt(s.pago)[5:]}**\n\n¿**Confirmo** tu pedido? (sí / no)")
        return self._crear_pedido(s)

    def _crear_pedido(self, s: Sesion) -> str:
        self._numero += 1
        envio = self._envio(s)
        p = Pedido(self._numero, [Linea(l.producto_id, l.cantidad, l.tamano, list(l.notas)) for l in s.carrito],
                   s.modalidad, s.pago, s.direccion, self.ahora(), s.subtotal, envio)
        s.pedidos.append(p)
        eta = self._eta(p.modalidad)
        txt = (f"🎉 **¡Pedido #{p.numero} confirmado!**\n" + self._lineas_texto(p.lineas) +
               f"\n\n**Total: {pesos(p.total)}** · {MODALIDADES[p.modalidad]} · {_pago_txt(p.pago)}")
        if p.direccion:
            txt += f"\n📍 {p.direccion}"
        txt += (f"\n⏱️ Tiempo estimado: **~{eta} min**.\n\nPuedes preguntarme «¿cómo va mi pedido?» cuando quieras. "
                "¡Gracias por elegirnos! 🍔")
        s.carrito, s.modalidad, s.pago, s.direccion, s.esperando, s.ultimos = [], None, None, None, None, []
        return txt

    # ══════════════ ESTADO DE PEDIDOS ══════════════
    @staticmethod
    def _eta(modalidad: str) -> int:
        return {"domicilio": NEGOCIO["tiempo_domicilio_min"], "llevar": NEGOCIO["tiempo_llevar_min"],
                "local": NEGOCIO["tiempo_local_min"]}[modalidad]

    def estado_pedido(self, p: Pedido) -> dict:
        eta = self._eta(p.modalidad)
        minutos = (self.ahora() - p.creado).total_seconds() / 60 * VELOCIDAD_SIMULACION
        if minutos < 1:
            cod, txt = "recibido", "✅ Pedido recibido"
        elif minutos < eta * 0.6:
            cod, txt = "cocina", "🍳 En preparación en cocina"
        elif minutos < eta:
            cod, txt = {"domicilio": ("camino", "🛵 En camino a tu dirección"),
                        "llevar": ("listo", "🛍️ Listo para recoger en el local"),
                        "local": ("servido", "🍽️ Saliendo a tu mesa")}[p.modalidad]
        else:
            cod, txt = "entregado", "🎉 Entregado / finalizado"
        return {"codigo": cod, "texto": txt, "restante": max(0, math.ceil(eta - minutos)), "eta": eta,
                "transcurrido": int(minutos), "progreso": min(100, int(100 * minutos / eta))}

    def _pedidos(self, texto: str, t: str, s: Sesion) -> str:
        m = re.search(r"\b(\d{3,6})\b", texto)
        pedido = None
        if m:
            num = int(m.group(1))
            pedido = next((p for p in s.pedidos if p.numero == num), None)
            if not pedido:
                return (f"🔎 No encuentro el pedido **#{num}** en esta conversación. Si lo hiciste por otro canal, "
                        f"llámanos al **{NEGOCIO['telefono']}** con tu número de orden y lo revisamos de inmediato.")
        elif s.pedidos:
            pedido = s.pedidos[-1]
        if not pedido:
            txt = "📦 Aún no tienes pedidos confirmados conmigo."
            txt += (" Tienes uno **en armado** que falta confirmar: escribe «confirmar pedido» 😉" if s.carrito
                    else " ¿Quieres que te ayude a armar uno?")
            return txt
        e = self.estado_pedido(pedido)
        txt = f"📦 **Pedido #{pedido.numero}**: {e['texto']}\n"
        if e["codigo"] == "entregado":
            return txt + "\n¡Esperamos que lo hayas disfrutado! Si algo salió mal, cuéntame y lo resolvemos."
        txt += (f"⏱️ Llevas {e['transcurrido']} min de ~{e['eta']} estimados; "
                f"faltan **~{e['restante']} min**.")
        if re.search(r"demor|tard|retras|mucho|lento|esper", t) and e["restante"] == 0:
            txt += (f"\n\n😔 Lamento la demora. Ya superamos el tiempo estimado; llámanos al "
                    f"**{NEGOCIO['telefono']}** para priorizar tu pedido.")
        elif re.search(r"demor|tard|retras|mucho|lento", t):
            txt += "\n\nSigue dentro del tiempo estimado, ¡gracias por tu paciencia! 🙏"
        return txt

    # ══════════════ ATENCIÓN AL CLIENTE ══════════════
    def _atencion(self, texto: str, t: str, s: Sesion) -> str:
        n = NEGOCIO
        reglas: list[tuple[str, Callable[[], str]]] = [
            (r"asesor|humano|persona real|agente|hablar con alguien|urgente|me equivoque",
             lambda: f"🙋 Con gusto te comunico con una persona: llama o escribe al **{n['telefono']}** "
                     f"(WhatsApp). Atendemos {n['horario'].lower()}"),
            (r"queja|reclam|mal servicio|pesim\w+|insatisf\w+|molest\w+|inconform\w+",
             lambda: self._queja(texto, s)),
            (r"felicit|excelente|buen servicio|muy rico|comentario positivo|sugerenc|sugerir|mejorar",
             lambda: "💛 ¡Muchas gracias por tu mensaje! Lo compartimos con el equipo. Si es una sugerencia, "
                     "cuéntame los detalles y la dejo anotada."),
            (r"reembolso|garantia|devol\w+",
             lambda: "🔁 Si tu pedido llegó con un problema (producto equivocado, incompleto o en mal estado), "
                     f"repórtalo dentro de las **2 horas** siguientes al **{n['telefono']}** y gestionamos "
                     "reposición o reembolso."),
            (r"reserva|cumpleanos|evento|mesa para",
             lambda: f"🎂 ¡Claro! Para reservas y celebraciones escríbenos o llámanos al **{n['telefono']}** "
                     "indicando fecha, hora y número de personas."),
            (r"promo|descuento|oferta|cupon",
             lambda: "🏷️ Hoy tenemos **2×1 en Perros Calientes Especiales** (demo) y domicilio gratis en pedidos "
                     f"desde **{pesos_(n['envio_gratis_desde'])}**."),
            (r"domicilio|envio|barrio|cobertura|reparten",
             lambda: f"🛵 ¡Sí hacemos domicilios! Costo: **{pesos_(n['envio_costo'])}** (gratis desde "
                     f"**{pesos_(n['envio_gratis_desde'])}**). Tiempo estimado: ~{n['tiempo_domicilio_min']} min. "
                     "Dime tu dirección al hacer el pedido y te confirmo cobertura."),
            (r"pago|pagar|tarjeta|nequi|daviplata|transferencia|efectivo|bono",
             lambda: f"💳 Aceptamos {n['pagos']}."),
            (r"horario|hora|abren|cierran|domingo|festivo|lunes|martes|sabado|atienden",
             lambda: f"🕒 {n['horario']}"),
            (r"ubicacion|direccion|donde|queda|llegar|ubicad",
             lambda: f"📍 Estamos en **{n['direccion']}**."),
            (r"telefono|whatsapp|contacto|correo|llamar|numero",
             lambda: f"📞 Puedes contactarnos al **{n['telefono']}** (llamadas y WhatsApp)."),
            (r"parqueadero|parqueo|estacionamiento",
             lambda: "🅿️ Sí, contamos con parqueadero para clientes (sujeto a disponibilidad)."),
            (r"pet|mascota|perro",
             lambda: "🐶 ¡Sí! Tenemos zona **pet friendly** en la terraza."),
            (r"factura",
             lambda: "🧾 Sí, emitimos **factura electrónica**. Dame tu correo y NIT/cédula al pagar."),
        ]
        for rx, f in reglas:
            if re.search(rx, t):
                return f() + "\n\n¿Te ayudo con algo más?"
        return (f"ℹ️ **Atención al cliente**\n- 🕒 {n['horario']}\n- 📍 {n['direccion']}\n- 💳 {n['pagos'].capitalize()}\n"
                f"- 📞 {n['telefono']}\n\n¿Tienes alguna queja, sugerencia o duda? Cuéntame 😊")

    def _queja(self, texto: str, s: Sesion) -> str:
        s.quejas.append(texto)
        return ("😔 Lamento mucho lo ocurrido. Dejé registrado tu mensaje para el equipo. Cuéntame **qué pasó y el "
                f"número de pedido** (si lo tienes) para ayudarte; si prefieres atención inmediata, llama al "
                f"**{NEGOCIO['telefono']}**.")

    # ══════════════ FUERA DE TEMA ══════════════
    def _fuera(self, texto: str, t: str, s: Sesion) -> str:
        if re.search(r"receta|preparar|cocinar|ingredientes para", t):
            intro = "👨‍🍳 Las recetas de la casa son secreto del chef, pero"
        else:
            intro = ["🤖 Eso se sale de mi cocina,", "😅 Uy, de eso no sé mucho:", "🍔 Mi especialidad es la comida,"][
                len(texto) % 3]
        return (f"{intro} puedo ayudarte con:\n- 🍽️ recomendaciones de platos\n- 📜 menú y precios\n"
                "- 🛒 tu pedido\n- ℹ️ horarios, pagos y ubicación\n\n¿Qué se te antoja hoy?")


def pesos_(v: int) -> str:
    return pesos(v)
