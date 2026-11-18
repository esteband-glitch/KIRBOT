"""Lanza la interfaz web de KIRBOT.   python app.py   (luego abre http://127.0.0.1:7860)"""
import argparse

from kirbot.clasificador import cargar_clasificador
from kirbot.dialogo import Kirbot
from kirbot.interfaz import crear_demo


def main():
    ap = argparse.ArgumentParser(description="KIRBOT – interfaz web")
    ap.add_argument("--modelo", choices=["auto", "transformer", "tfidf"], default="auto",
                    help="auto = Transformer si está entrenado; si no, TF-IDF de respaldo")
    ap.add_argument("--puerto", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="enlace público temporal de Gradio")
    ap.add_argument("--no-navegador", action="store_true")
    a = ap.parse_args()

    clf = cargar_clasificador(a.modelo)
    print(f"[KIRBOT] Clasificador activo: {clf.nombre}")
    demo = crear_demo(Kirbot(clf))
    demo.launch(server_port=a.puerto, share=a.share, inbrowser=not a.no_navegador, show_error=True)


if __name__ == "__main__":
    main()
