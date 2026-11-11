# Benchmark de decisores de evasion

## Por decisor

| decisor | ejecuciones | tasa_llegada_pct | avance_medio_m | t_efectivo_medio_s | decisiones | decisiones_aceptadas_pct | latencia_media_s | latencia_p95_s | coincide_con_regla_pct | pasos_peligrosos | motivos_fallo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev | 30 | 33.3 | 0.6 | 7.78 | 20 | 0.0 | 0.53 | 0.725 | 100.0 | 0 | jev_baja_confianza x20 |
| regla | 30 | 33.3 | 0.2 | 37.45 | 20 | 100.0 |  |  | 100.0 | 1003 | evasion_sin_salida x20 |
| sin_jev | 30 | 33.3 | 0.6 | 7.43 | 0 |  |  |  |  | 0 | jev_no_disponible x20 |

## Por decisor y ruta

| ruta | decisor | ejecuciones | tasa_llegada_pct | avance_medio_m | t_efectivo_medio_s | decisiones | decisiones_aceptadas_pct | latencia_media_s | latencia_p95_s | coincide_con_regla_pct | pasos_peligrosos | motivos_fallo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | jev | 10 | 0.0 | 1.27 | 11.92 | 10 | 0.0 | 0.52 | 0.632 | 100.0 | 0 | jev_baja_confianza x10 |
| 2 | jev | 10 | 100.0 | 0.49 | 10.04 | 0 |  |  |  |  | 0 |  |
| 3 | jev | 10 | 0.0 | 0.05 | 1.38 | 10 | 0.0 | 0.53 | 0.78 | 100.0 | 0 | jev_baja_confianza x10 |
| 1 | regla | 10 | 0.0 | 2.11 | 56.43 | 10 | 100.0 |  |  | 100.0 | 1003 | evasion_sin_salida x10 |
| 2 | regla | 10 | 100.0 | 0.49 | 10.04 | 0 |  |  |  |  | 0 |  |
| 3 | regla | 10 | 0.0 | -2.01 | 45.87 | 10 | 100.0 |  |  | 100.0 | 0 | evasion_sin_salida x10 |
| 1 | sin_jev | 10 | 0.0 | 1.27 | 11.4 | 0 |  |  |  |  | 0 | jev_no_disponible x10 |
| 2 | sin_jev | 10 | 100.0 | 0.49 | 10.04 | 0 |  |  |  |  | 0 |  |
| 3 | sin_jev | 10 | 0.0 | 0.05 | 0.85 | 0 |  |  |  |  | 0 | jev_no_disponible x10 |

## Mismo escenario, avance hacia el destino (empate si difiere < 0,05 m)

| comparacion | gana_primero | empates | gana_segundo |
| --- | --- | --- | --- |
| jev vs regla | 10 | 10 | 10 |
| jev vs sin_jev | 0 | 30 | 0 |
| regla vs sin_jev | 10 | 10 | 10 |
