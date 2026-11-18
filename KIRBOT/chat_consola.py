"""Chat de KIRBOT en la terminal (útil para depurar).   python chat_consola.py"""
import argparse

from kirbot.clasificador import cargar_clasificador
from kirbot.dialogo import BIENVENIDA, Kirbot, Sesion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["auto", "transformer", "tfidf"], default="auto")
    ap.add_argument("--debug", action="store_true", help="muestra intención, confianza y fuente")
    a = ap.parse_args()
    kb, ses = Kirbot(cargar_clasificador(a.modelo)), Sesion()
    print(f"[clasificador: {kb.clf.nombre}] escribe 'salir' para terminar\n\nKIRBOT> {BIENVENIDA}\n")
    while True:
        try:
            texto = input("Tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if texto.lower() in ("salir", "exit", "quit"):
            break
        r = kb.responder(texto, ses)
        print(f"\nKIRBOT> {r.texto}\n")
        if a.debug:
            print(f"   [intención={r.categoria} conf={r.confianza:.2f} fuente={r.fuente}]\n")


if __name__ == "__main__":
    main()
