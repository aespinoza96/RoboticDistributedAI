"""Construye el detector de personas del controlador: YOLOE-26s con la clase "person" fijada.

Por que: YOLO11n (COCO) casi no reconoce al peaton de Webots de perfil o de espaldas.
Medido sobre 39 imagenes del peaton (frente, perfil y espalda, de 1.3 a 3.8 m):
    YOLO11n 4/39 | YOLO-World "person" 14/39 | YOLO26s 17/39 | YOLOE-26s "person" 23/39 (umbral 0.15)

Fijar la clase y guardar el modelo deja un archivo de ~31 MB que el controlador carga sin el
modelo de texto (MobileCLIP, ~254 MB), que solo hace falta para construirlo.

    python3 mir100/controllers/keyboard_control/construir_detector_personas.py           # YOLOE-26s (recomendado)
    python3 mir100/controllers/keyboard_control/construir_detector_personas.py --world RUTA/yolov8s-worldv2.pt

La primera vez ultralytics descarga yoloe-26s-seg.pt y el modelo de texto desde los releases
oficiales de Ultralytics en GitHub. Los .pt resultantes no se suben al repo (ver .gitignore).
"""
import os
import sys

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))


def construir_yoloe():
    from ultralytics import YOLOE
    modelo = YOLOE("yoloe-26s-seg.pt")
    modelo.set_classes(["person"], modelo.get_text_pe(["person"]))
    salida = os.path.join(DIRECTORIO, "yoloe_personas.pt")
    modelo.save(salida)
    return salida


def construir_world(origen):
    from ultralytics import YOLO
    if not os.path.isfile(origen):
        raise FileNotFoundError("No existe {}".format(origen))
    modelo = YOLO(origen)
    modelo.set_classes(["person"])
    salida = os.path.join(DIRECTORIO, "yolo_personas.pt")
    modelo.save(salida)
    return salida


def main():
    try:
        salida = construir_world(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--world" else construir_yoloe()
    except Exception as error:  # noqa: BLE001 - mensaje claro para quien lo corre
        print("No se pudo construir el detector: {}: {}".format(type(error).__name__, error))
        return 1
    print("Detector de personas guardado en {}".format(salida))
    return 0


if __name__ == "__main__":
    sys.exit(main())
