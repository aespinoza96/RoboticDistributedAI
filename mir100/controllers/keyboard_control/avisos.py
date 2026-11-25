"""Avisos del MiR100 hacia las personas: color de las bandas de luz, pitido y voz.

- Luz: el PROTO Mir100 tiene el campo `color` (las bandas de luz del cuerpo, como en el
  robot real). El controlador es Supervisor y lo cambia en vivo; el PROTO no es plantilla,
  asi que cambiarlo no regenera el robot.
- Sonido y voz: se reproducen en el computador que corre la simulacion (macOS: `afplay`
  y `say`), sin bloquear el bucle de control. En otros sistemas, o con MIR_SONIDO=0, no
  suenan (la luz sigue funcionando).
"""
import os
import shutil
import subprocess
import sys

COLORES = {
    "normal": (0.0, 0.9, 1.0),     # cian, el color de fabrica del MiR100
    "persona": (1.0, 0.8, 0.0),    # amarillo: persona cerca, va con cuidado
    "apagado": (0.15, 0.12, 0.0),  # amarillo tenue, para el parpadeo
    "detenido": (1.0, 0.35, 0.0),  # naranja: no puede seguir por una persona
}
VOZ = os.getenv("MIR_VOZ", "Paulina")          # voz en espanol de macOS (es_MX)
PITIDO = "/System/Library/Sounds/Ping.aiff"


class Avisos:
    def __init__(self, campo_color=None):
        self.campo_color = campo_color
        self.color_actual = None
        self.con_sonido = (os.getenv("MIR_SONIDO", "1") != "0" and sys.platform == "darwin"
                           and shutil.which("say") is not None)
        self.voz_en_curso = None

    def luz(self, estado):
        """Cambia el color de las bandas solo si es distinto del actual."""
        if self.campo_color is None or estado == self.color_actual:
            return
        self.color_actual = estado
        self.campo_color.setSFColor(list(COLORES[estado]))

    def pitido(self):
        self._lanzar(["afplay", PITIDO])

    def decir(self, texto):
        """Habla sin bloquear; si todavia esta hablando, no encima otra frase."""
        if self.voz_en_curso is not None and self.voz_en_curso.poll() is None:
            return
        self.voz_en_curso = self._lanzar(["say", "-v", VOZ, texto])

    def _lanzar(self, comando):
        if not self.con_sonido:
            return None
        try:
            return subprocess.Popen(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.con_sonido = False
            return None
