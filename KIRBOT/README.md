# 🍔 KIRBOT – Asistente virtual para restaurantes

Chatbot en español: clasificador de intenciones (Transformer o TF‑IDF) + capa de reglas + lógica de diálogo con
memoria de sesión + interfaz web (Gradio). Los datos del menú y del negocio son **ficticios** (de demostración).

## Puesta en marcha en Visual Studio Code
1. Instala **Python 3.10–3.12** y en VS Code las extensiones *Python* y *Jupyter* (te las sugiere al abrir la carpeta).
2. `File ▸ Open Folder…` → selecciona esta carpeta (`KIRBOT/`).
3. Terminal de VS Code:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (Si solo quieres probar sin Transformer, basta instalar las líneas hasta `pytest`; `torch`/`transformers` son opcionales.)
4. Ejecuta (o usa **Run and Debug** ▸ configuraciones "KIRBOT · …"):
   ```bash
   python app.py                 # interfaz web → http://127.0.0.1:7860
   python chat_consola.py --debug  # chat en terminal mostrando intención y confianza
   pytest -q                     # pruebas automáticas
   python entrenar.py --modelo transformer --epocas 10   # entrena DistilBERT (guarda en modelos/)
   ```
5. Para el cuaderno: abre `KIRBOT.ipynb`, elige el kernel `.venv` y ejecuta las celdas.

Sin Transformer entrenado, KIRBOT usa automáticamente el clasificador **TF‑IDF** (se entrena en ~1 s).

## Estructura
| Ruta | Qué hace |
|---|---|
| `data/ejemplos.csv` | Dataset (210 frases, 6 categorías). **Aquí agregas ejemplos** para mejorar el modelo |
| `kirbot/config.py` | Datos del negocio (horario, dirección, pagos, costos de envío, tiempos) |
| `kirbot/menu.py` | Menú, alias de productos, cantidades, plurales |
| `kirbot/reglas.py` | Señales de texto que refuerzan al modelo |
| `kirbot/clasificador.py` | TF‑IDF y Transformer, evaluación |
| `kirbot/dialogo.py` | Respuestas, carrito, checkout, estado de pedidos |
| `kirbot/interfaz.py` | Interfaz Gradio |
| `tests/` | Pruebas |

## Qué mejoró en las respuestas
- **Recomendaciones filtradas** (pollo, vegetariano, picante, niños, ligero, "sin cebolla", barato, combo, maridaje de bebidas).
- **Menú**: precio, ingredientes, tamaños de pizza, categorías, adicionales, respuestas por producto.
- **Pedidos reales**: cantidades ("dos"), notas por producto, quitar/cambiar, aclaración de opciones ambiguas ("una malteada" → ¿chocolate o vainilla?), pizza familiar, checkout por pasos (modalidad → dirección → pago → confirmación), número de pedido y costo de domicilio.
- **Seguimiento** por tiempo (usa `KIRBOT_VELOCIDAD=30` para verlo avanzar rápido en demos).
- **Atención**: 14 temas (horario, pagos, ubicación, queja, reserva, factura, mascotas…).
- **Contexto**: "agrégala", "la segunda", "sí" tras una recomendación.
- Si la confianza es baja, **pregunta** en lugar de inventar; saludos/gracias/despedida se resuelven sin modelo.

## Cómo seguir mejorándolo
1. Agrega frases reales de clientes a `data/ejemplos.csv` (con tildes, sin tildes, con errores y jerga) y reentrena.
2. Revisa los errores que imprime `entrenar.py` y `chat_consola.py --debug` para ver qué intención falló.
3. Reemplaza los datos de `config.py` y `menu.py` por los reales (y conecta pedidos/quejas a tu sistema: hoy se guardan solo en memoria de la sesión).
