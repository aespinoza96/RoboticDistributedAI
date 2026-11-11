# Resume experimentos/resultados/resultados.jsonl en tablas de benchmark.
# Uso (desde la raiz del repo):  python3 experimentos/analizar.py
#
# Genera en experimentos/resultados/:
#   resumen_por_decisor.csv  una fila por decisor
#   resumen_por_ruta.csv     una fila por decisor y ruta
#   decisiones.csv           cada decision de evasion (decisor, propuesta, latencia, aceptada...)
#   comparacion_pareada.csv  mismo escenario (ruta, rep) con cada decisor
#   benchmark.md             las tablas anteriores en markdown
#   benchmark.png            avance medio hacia el destino y tasa de llegada por decisor

import csv
import json
import os
import statistics
from collections import Counter, defaultdict

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
ENTRADA = os.path.join(CARPETA, "resultados.jsonl")
ORDEN = ["jev", "regla", "sin_jev"]


def media(valores):
    return round(statistics.mean(valores), 2) if valores else None


def percentil(valores, p):
    if not valores:
        return None

    ordenados = sorted(valores)
    indice = min(len(ordenados) - 1, int(round(p / 100 * (len(ordenados) - 1))))

    return round(ordenados[indice], 3)


def avance(r):
    return r["distancia_inicial_m"] - r["distancia_final_m"]


def resumir(filas):
    llegadas = [r for r in filas if r["resultado"] == "llegada"]
    decisiones = [d for r in filas for d in r["decisiones"]]
    latencias = [d["latencia_s"] for d in decisiones if d["decisor"] == "jev" and d["propuesta"] is not None]
    comparables = [d for d in decisiones if d["propuesta"] is not None and d["regla_habria_elegido"] is not None]
    motivos = Counter(r["motivo"] or r["resultado"] for r in filas if r["resultado"] != "llegada")

    return {
        "ejecuciones": len(filas),
        "llegadas": len(llegadas),
        "tasa_llegada_pct": round(100 * len(llegadas) / len(filas), 1) if filas else None,
        "avance_medio_m": media([avance(r) for r in filas]),
        "distancia_final_media_m": media([r["distancia_final_m"] for r in filas]),
        "t_efectivo_medio_llegadas_s": media([r["t_efectivo_s"] for r in llegadas]),
        "t_efectivo_medio_s": media([r["t_efectivo_s"] for r in filas]),
        "recorrido_medio_m": media([r["recorrido_m"] for r in filas]),
        "distancia_minima_lidar_m": min((r["distancia_minima_lidar_m"] for r in filas if r["distancia_minima_lidar_m"] is not None), default=None),
        "pasos_peligrosos": sum(r["pasos_peligrosos"] for r in filas),
        "decisiones": len(decisiones),
        "decisiones_aceptadas_pct": round(100 * sum(d["aceptada"] for d in decisiones) / len(decisiones), 1) if decisiones else None,
        "latencia_media_s": media(latencias),
        "latencia_p95_s": percentil(latencias, 95),
        "espera_total_decisiones_s": round(sum(d["latencia_s"] for d in decisiones), 2),
        "coincide_con_regla_pct": round(100 * sum(d["propuesta"] == d["regla_habria_elegido"] for d in comparables) / len(comparables), 1) if comparables else None,
        "motivos_fallo": "; ".join("{} x{}".format(k, v) for k, v in motivos.most_common()),
    }


def escribir_csv(path, filas):
    if not filas:
        return

    with open(path, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def tabla_md(filas, columnas):
    lineas = ["| " + " | ".join(columnas) + " |", "| " + " | ".join("---" for _ in columnas) + " |"]

    for fila in filas:
        lineas.append("| " + " | ".join("" if fila.get(c) is None else str(fila.get(c)) for c in columnas) + " |")

    return "\n".join(lineas)


def main():
    with open(ENTRADA, encoding="utf-8") as archivo:
        registros = [json.loads(linea) for linea in archivo if linea.strip()]

    decisores = [d for d in ORDEN if any(r["decisor"] == d for r in registros)]

    por_decisor = [dict(decisor=d, **resumir([r for r in registros if r["decisor"] == d])) for d in decisores]

    por_ruta = []
    for d in decisores:
        for ruta in sorted({r["ruta"] for r in registros}):
            filas = [r for r in registros if r["decisor"] == d and r["ruta"] == ruta]
            if filas:
                por_ruta.append(dict(decisor=d, ruta=ruta, **resumir(filas)))

    decisiones = [dict(decisor_ejecucion=r["decisor"], ruta=r["ruta"], rep=r["rep"], **d)
                  for r in registros for d in r["decisiones"]]

    # Comparacion pareada: el mismo escenario (ruta, rep) con cada decisor
    escenarios = defaultdict(dict)
    for r in registros:
        escenarios[(r["ruta"], r["rep"])][r["decisor"]] = r

    pareada = []
    for (ruta, rep), por in sorted(escenarios.items()):
        fila = {"ruta": ruta, "rep": rep}
        for d in decisores:
            r = por.get(d)
            fila[d + "_resultado"] = r["resultado"] if r else None
            fila[d + "_avance_m"] = round(avance(r), 2) if r else None
        pareada.append(fila)

    escribir_csv(os.path.join(CARPETA, "resumen_por_decisor.csv"), por_decisor)
    escribir_csv(os.path.join(CARPETA, "resumen_por_ruta.csv"), por_ruta)
    escribir_csv(os.path.join(CARPETA, "decisiones.csv"), decisiones)
    escribir_csv(os.path.join(CARPETA, "comparacion_pareada.csv"), pareada)

    # Victorias por pares en avance hacia el destino (mismo escenario)
    pares = []
    for i, a in enumerate(decisores):
        for b in decisores[i + 1:]:
            gana_a = gana_b = empate = 0
            for por in escenarios.values():
                if a in por and b in por:
                    diferencia = avance(por[a]) - avance(por[b])
                    if abs(diferencia) < 0.05:
                        empate += 1
                    elif diferencia > 0:
                        gana_a += 1
                    else:
                        gana_b += 1
            pares.append({"comparacion": "{} vs {}".format(a, b), "gana_" + "primero": gana_a, "empates": empate, "gana_segundo": gana_b})

    columnas = ["decisor", "ejecuciones", "tasa_llegada_pct", "avance_medio_m", "t_efectivo_medio_s", "decisiones",
                "decisiones_aceptadas_pct", "latencia_media_s", "latencia_p95_s", "coincide_con_regla_pct",
                "pasos_peligrosos", "motivos_fallo"]

    with open(os.path.join(CARPETA, "benchmark.md"), "w", encoding="utf-8") as archivo:
        archivo.write("# Benchmark de decisores de evasion\n\n")
        archivo.write("## Por decisor\n\n" + tabla_md(por_decisor, columnas) + "\n\n")
        archivo.write("## Por decisor y ruta\n\n" + tabla_md(por_ruta, ["ruta"] + columnas) + "\n\n")
        archivo.write("## Mismo escenario, avance hacia el destino (empate si difiere < 0,05 m)\n\n"
                      + tabla_md(pares, ["comparacion", "gana_primero", "empates", "gana_segundo"]) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figura, ejes = plt.subplots(1, 2, figsize=(9, 3.6))
        ejes[0].bar(decisores, [f["avance_medio_m"] for f in por_decisor], color="#4c72b0")
        ejes[0].set_title("Avance medio hacia el destino (m)")
        ejes[1].bar(decisores, [f["tasa_llegada_pct"] for f in por_decisor], color="#55a868")
        ejes[1].set_title("Tasa de llegada (%)")
        ejes[1].set_ylim(0, 100)
        figura.tight_layout()
        figura.savefig(os.path.join(CARPETA, "benchmark.png"), dpi=150)
    except ImportError:
        pass

    print(open(os.path.join(CARPETA, "benchmark.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
