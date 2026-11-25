"""Tests de las opciones por orden del controlador (parse_task_options), sin Webots."""
import ast
import os
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA = os.path.join(RAIZ, "mir100", "controllers", "keyboard_control", "keyboard_control.py")


def cargar():
    with open(RUTA, encoding="utf-8") as archivo:
        arbol = ast.parse(archivo.read())
    nombres_constantes = {"PLAN_VELOCIDAD", "OPCIONES_AL_VER_PERSONA", "DISTANCIA_PERSONA_DEFECTO",
                          "VELOCIDAD_MIN", "VELOCIDAD_MAX_ORDEN"}
    nodos = [n for n in arbol.body if isinstance(n, ast.Assign) and any(
        (isinstance(t, ast.Name) and t.id in nombres_constantes) or
        (isinstance(t, ast.Tuple) and any(getattr(e, "id", None) in nombres_constantes for e in t.elts)) for t in n.targets)]
    nodos += [n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name in ("parse_task_options", "option_args")]
    espacio = {}
    exec(compile(ast.Module(nodos, []), RUTA, "exec"), espacio)
    return espacio


class TestOpciones(unittest.TestCase):
    def setUp(self):
        self.e = cargar()

    def test_por_defecto(self):
        o = self.e["parse_task_options"]({"destino": "ENTREGA_1"})
        self.assertEqual(o["al_ver_persona"], "rodear")
        self.assertEqual(self.e["option_args"](o), {})

    def test_detenerse_y_limites(self):
        o = self.e["parse_task_options"]({"al_ver_persona": "Detenerse", "distancia_persona": 50, "velocidad": 2})
        self.assertEqual(o["al_ver_persona"], "detenerse")
        self.assertEqual(o["distancia_persona"], 10.0)
        self.assertEqual(o["velocidad"], self.e["VELOCIDAD_MAX_ORDEN"])
        self.assertEqual(set(self.e["option_args"](o)), {"al_ver_persona", "distancia_persona", "velocidad"})

    def test_opcion_invalida(self):
        with self.assertRaises(ValueError):
            self.e["parse_task_options"]({"al_ver_persona": "saludar"})


if __name__ == "__main__":
    unittest.main()
