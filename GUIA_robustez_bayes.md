# Pruebas de robustez de M0 y M1: Epsilon Eridani

## Alcance de esta ampliación

Este paquete amplía `src/model_data.py` **sin eliminar** las funciones anteriores de análisis exploratorio, los diagnósticos ni las ejecuciones originales M0/M1/M2. Añade el ejecutor independiente `ejecutar_robustez_bayes.py` para Spyder.

**Pregunta científica:** ¿sigue siendo favorable la componente orbital de M1 cuando (i) no se dispone de APF, (ii) relajamos los límites inferiores de los priors del ruido correlacionado y (iii) repetimos el muestreo con otras semillas?

El ajuste no se hace instrumento a instrumento. Cada escenario utiliza **una sola serie conjunta** con parámetros físicos compartidos y offsets/jitters de cada instrumento conservado.

### Qué mantenemos constante

- **M0:** tendencia lineal global, offsets y jitters por instrumento, `RotationTerm` para rotación y `Matern32Term` para variabilidad lenta.
- **M1:** exactamente M0 más una órbita **circular** con periodo, semiamplitud y fase.
- Para cada par M0/M1 se usan idénticos datos, priors de ruido y parámetros de `dynesty` (`nlive`, `dlogz`, semilla).
- El prior orbital de M1 permanece **log-uniforme en 200–6000 días**, con semiamplitud uniforme en 0–80 m/s y fase uniforme en 0–1.
- El índice de actividad S_MW no forma parte de la función de verosimilitud de este ajuste de RV. Sigue siendo una comprobación externa.

## Escenarios disponibles

| `ESCENARIO` | Observaciones | Modificaciones respecto al original | Qué investiga |
|---|---|---|---|
| `SIN_APF` | 1.336 RV, 6 instrumentos | Excluye APF; priors originales | ¿La preferencia por M1 depende críticamente de APF? |
| `Q0_AMPLIADO` | 1.888 RV, 7 instrumentos | `gp_Q0` log-uniforme 0,03–20, antes 0,2–20 | ¿El límite inferior de Q0 era restrictivo? |
| `ESCALA_AMPLIADA` | 1.888 RV, 7 instrumentos | `gp_escala_lenta_d` log-uniforme 30–5000, antes 150–5000 | ¿La escala lenta estaba forzada a superar 150 días? |
| `BASE_REPETICION` | 1.888 RV, 7 instrumentos | Priors originales, nueva semilla | ¿Se reproduce la comparación M0/M1 anterior con otro muestreo? |
| `BASE_SEMILLA_2` | 1.888 RV, 7 instrumentos | Priors originales, segunda semilla | ¿Es estable el resultado entre ejecuciones? |
| `SIN_APF_GP_AMPLIADO` | 1.336 RV, 6 instrumentos | Excluye APF y amplía ambos priors | Sensibilidad combinada, opcional |

**Motivo de variar los priors por separado:** si se modifica a la vez `Q0` y la escala lenta no se sabe cuál de ellos provoca el cambio. Por eso las pruebas `Q0_AMPLIADO` y `ESCALA_AMPLIADA` son separadas.

## Ejecución recomendada

En `ejecutar_robustez_bayes.py`:

MODO = "PRUEBA"
ESCENARIO = "SIN_APF"
NLIVE = 400
DLOGZ = 0.05
SEMILLA = 31415
SEMILLA_ALTERNATIVA = 27182
PERIODO_MIN = 200.0
PERIODO_MAX = 6000.0

1. Ejecuta `MODO = "PRUEBA"`. Solo evalúa dos funciones de verosimilitud y comprueba que sean finitas; **NO calcula evidencias**, **NO es una inferencia**.
2. Cambia `MODO = "AMBOS"` y conserva `ESCENARIO = "SIN_APF"`. Se ajustan M0 y M1 sobre las 1.336 RV conservadas.
3. Cuando finalice, cambia `ESCENARIO = "Q0_AMPLIADO"`; mantiene `MODO = "AMBOS"` y ejecuta.
4. Repite con `ESCENARIO = "ESCALA_AMPLIADA"`.
5. Si las tres pruebas funcionan, ejecuta también `BASE_REPETICION` y `BASE_SEMILLA_2` para estudiar estabilidad de la evidencia respecto al muestreo. La prueba combinada `SIN_APF_GP_AMPLIADO` es opcional.
6. En cualquier momento, `MODO = "RESUMEN"` lee lo ya completado sin lanzar ajustes nuevos.

Si necesitas interrumpir una ejecución, el archivo `checkpoint.dynesty` queda guardado. Al volver a ejecutar la **misma combinación** de escenario, `NLIVE`, `DLOGZ` y semilla, se reanuda. No cambies estos valores en una carpeta que ya tiene un checkpoint.

Si ejecutas solo `MODO = "M0"`, el programa guarda M0; ejecuta luego `MODO = "M1"` con la misma configuración para completar la comparación. `MODO = "AMBOS"` automatiza ese orden.

## Estructura de resultados

```text
TCB_PR2/
  results/
    modelado_global/
      bayes_GP_kepler/        # M0, M1, M2 ORIGINALES: NO se tocan
      robustez_bayes/        # NUEVO
        SIN_APF/
          M0/
          M1/
          comparacion_M1_vs_M0.csv
        Q0_AMPLIADO/
          M0/
          M1/
          comparacion_M1_vs_M0.csv
        ESCALA_AMPLIADA/
          M0/
          M1/
          comparacion_M1_vs_M0.csv
        BASE_REPETICION/
          M0/
          M1/
        BASE_SEMILLA_2/
          M0/
          M1/
        resumen_robustez.csv
        resumen_robustez.png
        resumen_robustez.pdf
```

Cada M0/M1 nuevo genera `configuracion.json`, `evidencia.csv`, `parametros.csv`, `muestras_posteriores.csv`, `resultado_pesado.npz`, histogramas posteriores, componentes y checkpoints.

## Qué comparar para obtener conclusiones

La comparación válida es **dentro de cada escenario**:

\[
\Delta\ln Z_{10}=\ln Z(M1)-\ln Z(M0),
\qquad
\sigma_{\Delta}\simeq\sqrt{\sigma_{\ln Z,M0}^2+\sigma_{\ln Z,M1}^2}.
\]

- Si `SIN_APF` sigue mostrando \(\Delta\ln Z_{10}\) claramente positivo, la preferencia por una órbita **no desaparecería** al retirar APF. Comprueba también si el periodo \(P\) permanece cerca del máximo global anterior y si su posterior es multimodal.
- Si la evidencia favorable al planeta se pierde o el periodo es inestable, la interpretación debe matizarse: puede haber degeneración con la ventana temporal, la cobertura instrumental o el ruido estelar.
- Si ampliar el prior de `gp_Q0` o de `gp_escala_lenta_d` modifica fuertemente \(\Delta\ln Z\), las conclusiones son **sensibles a las hipótesis de ruido**. No se debe interpretar el resultado como prueba definitiva.
- Si las dos repeticiones de base difieren más de lo esperable por los errores de `lnZ`, necesitaremos mayor resolución o un método de exploración más robusto.

**No hay que comparar directamente `lnZ(M1, SIN_APF)` con `lnZ(M1, BASE_REPETICION)`**: son evidencias de dos datasets diferentes. Tampoco confundas el error de Nested Sampling de `lnZ` con la incertidumbre de modelo astrofísico. Los percentiles 16/50/84 representan incertidumbre posterior **condicionada al modelo y al prior**.

### Precisiones para la memoria

Este análisis conserva dos limitaciones del código original/refe:

1. Para los offsets se utilizan priors gaussianos propios centrados sobre medianas instrumentales observadas. Esta elección es parcialmente informada por los datos (*Bayes*) y la evidencia está condicionada a ella.
2. El GP se ajusta sobre las RV y no conjuntamente con S_MW. Una señal residual compartida con actividad puede seguir afectando la inferencia orbital.