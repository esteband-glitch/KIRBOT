"""Interfaz web (Gradio) de KIRBOT."""
from __future__ import annotations

import html
import inspect

import gradio as gr

from .dialogo import BIENVENIDA, MODALIDADES, Kirbot, Respuesta, Sesion
from .menu import pesos

ACCESOS_RAPIDOS = [
    ("🍽️ Recomiéndame algo", "Hola, no sé qué pedir. ¿Qué me recomiendas?"),
    ("🌶️ Algo picante", "Qué me ofreces si me gustan los sabores picantes"),
    ("📜 Ver el menú", "Quiero consultar el menú completo"),
    ("🛒 Pedir hamburguesa + gaseosa", "Quiero una hamburguesa clásica y una gaseosa"),
    ("📦 ¿Cómo va mi pedido?", "Quiero saber el estado de mi pedido"),
    ("🕒 Horarios y ubicación", "Cuáles son los horarios y dónde están ubicados"),
]

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
#hero {background: linear-gradient(135deg,#b91c1c,#ea580c 70%,#f59e0b); border-radius: 20px; padding: 24px 30px;
       color: #fff; box-shadow: 0 10px 28px rgba(185,28,28,.25); margin-bottom: 12px;}
#hero h1 {margin: 0 0 4px; font-size: 30px; letter-spacing: -.5px; color: #fff;}
#hero p {margin: 0; color: #fef3c7; line-height: 1.5;}
#hero .tag {display:inline-block; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 8px;
            background: rgba(255,255,255,.18); padding: 3px 10px; border-radius: 999px;}
.panel-k {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 14px 16px;
          background: var(--block-background-fill);}
.panel-k h4 {margin: 0 0 8px; font-size: 15px;}
.panel-k ul {margin: 0; padding-left: 18px;}
.panel-k li {margin: 3px 0;}
.total-k {margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border-color-primary); font-weight: 700;}
.muted-k {opacity: .7; font-size: 13px;}
.nota-k {opacity: .75; font-style: italic;}
.bar-k {height: 8px; background: #fed7aa; border-radius: 99px; overflow: hidden; margin: 6px 0 2px;}
.bar-k > div {height: 100%; background: linear-gradient(90deg,#ea580c,#dc2626); border-radius: 99px;
              transition: width .6s ease;}
.pedido-k {margin-bottom: 12px;}
.dark .bar-k {background: #7c2d12;}
footer {visibility: hidden;}
"""


GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


def _tema():
    return gr.themes.Soft(primary_hue="orange", neutral_hue="stone",
                          font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"])


def _crear_chatbot(inicial):
    """Chatbot compatible con Gradio 5 y 6 (los parámetros cambiaron entre versiones)."""
    params = inspect.signature(gr.Chatbot.__init__).parameters
    kw = dict(value=inicial, height=520, show_label=False, render_markdown=True, elem_id="chat")
    if "type" in params:
        kw["type"] = "messages"
    if "show_copy_button" in params:
        kw["show_copy_button"] = True
    elif "buttons" in params:
        kw["buttons"] = ["copy"]
    return gr.Chatbot(**kw)


def lanzar(demo: gr.Blocks, **kwargs):
    """Lanza la app. En Gradio 6 el tema y el CSS se pasan a launch(); en Gradio 5 ya van en Blocks."""
    if GRADIO_6:
        kwargs.setdefault("theme", _tema())
        kwargs.setdefault("css", CSS)
    return demo.launch(**kwargs)


def html_carrito(s: Sesion, kb: Kirbot) -> str:
    if not s.carrito:
        return ('<div class="panel-k"><h4>🛒 Tu pedido</h4><div class="muted-k">Aún no has agregado nada. '
                'Pídeme algo o usa los accesos rápidos 😊</div></div>')
    filas = ""
    for l in s.carrito:
        nota = f' <span class="nota-k">({html.escape(", ".join(l.notas))})</span>' if l.notas else ""
        filas += f"<li>{l.cantidad} × {html.escape(l.nombre)}{nota} — <b>{pesos(l.subtotal)}</b></li>"
    envio = kb._envio(s)
    extra = ""
    if s.modalidad:
        extra += f'<div class="muted-k">Modalidad: {MODALIDADES[s.modalidad]}</div>'
    if s.modalidad == "domicilio":
        extra += f'<div class="muted-k">Domicilio: {"gratis 🎉" if envio == 0 else pesos(envio)}</div>'
    if s.direccion:
        extra += f'<div class="muted-k">📍 {html.escape(s.direccion)}</div>'
    if s.pago:
        extra += f'<div class="muted-k">💳 {html.escape(s.pago)}</div>'
    return (f'<div class="panel-k"><h4>🛒 Tu pedido</h4><ul>{filas}</ul>{extra}'
            f'<div class="total-k">Total: {pesos(s.subtotal + envio)}</div></div>')


def html_pedidos(s: Sesion, kb: Kirbot) -> str:
    if not s.pedidos:
        return ('<div class="panel-k"><h4>📦 Seguimiento</h4><div class="muted-k">Cuando confirmes un pedido '
                'podrás seguirlo aquí en tiempo real.</div></div>')
    bloques = ""
    for p in reversed(s.pedidos[-3:]):
        e = kb.estado_pedido(p)
        resto = "" if e["codigo"] == "entregado" else f' · faltan ~{e["restante"]} min'
        bloques += (f'<div class="pedido-k"><b>Pedido #{p.numero}</b> · {pesos(p.total)}<br>{e["texto"]}{resto}'
                    f'<div class="bar-k"><div style="width:{e["progreso"]}%"></div></div></div>')
    return f'<div class="panel-k"><h4>📦 Seguimiento</h4>{bloques}</div>'


def _detalle(r: Respuesta | None) -> tuple[str, dict]:
    if r is None:
        return "Aún no hay mensajes.", {}
    probs = dict(sorted(r.probabilidades.items(), key=lambda x: -x[1])[:3]) or {r.categoria: float(r.confianza)}
    return (f"**Intención final:** `{r.categoria}` · **confianza:** {r.confianza:.0%} · **fuente:** `{r.fuente}`",
            probs)


def crear_demo(kb: Kirbot) -> gr.Blocks:
    inicial = [{"role": "assistant", "content": BIENVENIDA}]

    opciones = {} if GRADIO_6 else {"css": CSS, "theme": _tema()}
    with gr.Blocks(title="KIRBOT | Asistente de Restaurante", **opciones) as demo:
        gr.HTML(f'<div id="hero"><div class="tag">PROYECTO ACADÉMICO · NLP</div>'
                f'<h1>🍔 KIRBOT · Asistente de Restaurantes</h1>'
                f'<p>Recomendaciones, menú, pedidos y atención al cliente con un clasificador de intenciones '
                f'({html.escape(kb.clf.nombre)}) + lógica de diálogo.</p></div>')

        sesion = gr.State(Sesion())
        with gr.Row(equal_height=False):
            with gr.Column(scale=7):
                chat = _crear_chatbot(inicial)
                with gr.Row():
                    entrada = gr.Textbox(placeholder="Escribe aquí… ej.: «dos hamburguesas clásicas sin cebolla»",
                                         show_label=False, lines=1, max_length=500, scale=8, container=False,
                                         autofocus=True)
                    enviar = gr.Button("Enviar ➤", variant="primary", scale=1, min_width=110)
                with gr.Accordion("🔍 Detalles del modelo (último mensaje)", open=False):
                    detalle_md = gr.Markdown("Aún no hay mensajes.")
                    detalle_lbl = gr.Label(num_top_classes=3, show_label=False)
            with gr.Column(scale=4):
                carrito = gr.HTML(html_carrito(Sesion(), kb))
                pedidos = gr.HTML(html_pedidos(Sesion(), kb))
                gr.Markdown("#### 🚀 Accesos rápidos")
                botones = [gr.Button(etq, size="sm") for etq, _ in ACCESOS_RAPIDOS]
                reiniciar = gr.Button("↺ Reiniciar conversación", variant="secondary")

        salidas = [chat, sesion, entrada, carrito, pedidos, detalle_md, detalle_lbl]

        def hablar(mensaje, historial, ses):
            historial = list(historial or [])
            mensaje = (mensaje or "").strip()
            if not mensaje:
                md, lbl = _detalle(None)
                return historial, ses, "", html_carrito(ses, kb), html_pedidos(ses, kb), md, lbl
            r = kb.responder(mensaje, ses)
            historial += [{"role": "user", "content": mensaje}, {"role": "assistant", "content": r.texto}]
            md, lbl = _detalle(r)
            return historial, ses, "", html_carrito(ses, kb), html_pedidos(ses, kb), md, lbl

        def limpiar():
            ses = Sesion()
            md, lbl = _detalle(None)
            return inicial, ses, "", html_carrito(ses, kb), html_pedidos(ses, kb), md, lbl

        enviar.click(hablar, [entrada, chat, sesion], salidas, api_name="enviar")
        entrada.submit(hablar, [entrada, chat, sesion], salidas, api_name="enviar_enter")
        reiniciar.click(limpiar, None, salidas, api_name="reiniciar")
        for i, (btn, (_, frase)) in enumerate(zip(botones, ACCESOS_RAPIDOS)):
            btn.click(lambda h, s_, q=frase: hablar(q, h, s_), [chat, sesion], salidas, api_name=f"acceso_{i + 1}")

        # el estado del pedido avanza con el tiempo: refresca el panel cada 5 s
        reloj = gr.Timer(5)
        reloj.tick(lambda s_: html_pedidos(s_, kb), [sesion], [pedidos], show_progress="hidden", api_name="refrescar_pedidos")
    return demo