import json
import pandas as pd
from collections import Counter
from textstat import fernandez_huerta

from analysis.pregunta_respuesta import detectar_pregunta, detectar_respuesta
from analysis.interrupciones import detectar_interrupcion
from analysis.emociones import detectar_emocion
from analysis.palabras_clave import detectar_palabras_clave

CONECTORES = {
    "además", "entonces", "o sea", "bueno", "así que", "por lo tanto", "por eso",
    "en realidad", "claro", "digamos", "también", "sin embargo", "aunque"
}

PREGUNTAS_ABIERTAS = {"qué", "cómo", "cuándo", "dónde", "por qué", "cuál", "cuáles"}
PALABRAS_ACCION = {"ahorrar", "resolver", "lograr", "mejorar", "ganar", "acceder", "beneficio", "ahorro", "obtener"}

def es_pregunta_abierta(texto):
    return any(texto.startswith(w + " ") for w in PREGUNTAS_ABIERTAS)

def procesar_semantico(transcripcion_path):
    with open(transcripcion_path, "r", encoding="utf-8") as f:
        segmentos = json.load(f)

    resultados = []
    seg_anterior = None

    for seg in segmentos:
        texto = seg.get('text', '').strip().lower()

        es_preg = detectar_pregunta(texto)
        es_resp = detectar_respuesta(texto)
        interrupcion = detectar_interrupcion(seg, seg_anterior)
        emocion = detectar_emocion(texto)
        palabras_clave = detectar_palabras_clave(texto)
        legibilidad = fernandez_huerta(texto) if texto else None

        palabras = [p for p in texto.split() if len(p) > 2]
        conteo = Counter(palabras)
        repeticiones = sum(1 for _, c in conteo.items() if c >= 2)

        conectores_detectados = [c for c in CONECTORES if c in texto]
        uso_excesivo_conectores = len(conectores_detectados) > 3

        resultados.append({
            'speaker': seg['speaker'],
            'duracion': round(seg['end'] - seg['start'], 2),
            'texto': texto,
            'es_pregunta': es_preg,
            'es_pregunta_abierta': es_preg and es_pregunta_abierta(texto),
            'es_respuesta': es_resp,
            'interrupcion': interrupcion,
            'emocion_detectada': emocion,
            'indice_legibilidad': legibilidad,
            'num_repeticiones': repeticiones,
            'uso_excesivo_conectores': uso_excesivo_conectores,
            'palabras_totales': len(palabras),
            'palabras_unicas': len(set(palabras)),
            'palabras_accion': sum(p in PALABRAS_ACCION for p in palabras)
        })

        seg_anterior = seg

    df = pd.DataFrame(resultados)

    resumenes = []
    for speaker, grupo in df.groupby("speaker"):
        palabras = " ".join(grupo["texto"]).split()
        top_palabras = [p for p, _ in Counter(palabras).most_common(5) if len(p) > 2]

        total_preguntas = grupo["es_pregunta"].sum()
        preguntas_abiertas = grupo["es_pregunta_abierta"].sum()
        pct_abiertas = (preguntas_abiertas / total_preguntas * 100) if total_preguntas > 0 else 0

        total_palabras = grupo["palabras_totales"].sum()
        total_unicas = grupo["palabras_unicas"].sum()
        diversidad = (total_unicas / total_palabras) if total_palabras > 0 else 0

        resumen = {
            f"{speaker}_duracion_total": grupo["duracion"].sum(),
            f"{speaker}_num_preguntas": total_preguntas,
            f"{speaker}_num_preguntas_abiertas": preguntas_abiertas,
            f"{speaker}_pct_preguntas_abiertas": round(pct_abiertas, 2),
            f"{speaker}_num_respuestas": grupo["es_respuesta"].sum(),
            f"{speaker}_num_interrupciones": grupo["interrupcion"].sum(),
            f"{speaker}_emocion_predominante": grupo["emocion_detectada"].mode().iloc[0] if not grupo["emocion_detectada"].mode().empty else None,
            f"{speaker}_media_legibilidad": grupo["indice_legibilidad"].mean(),
            f"{speaker}_total_repeticiones": grupo["num_repeticiones"].sum(),
            f"{speaker}_uso_excesivo_conectores": grupo["uso_excesivo_conectores"].sum(),
            f"{speaker}_palabras_comunes": ", ".join(top_palabras),
            f"{speaker}_diversidad_lexica": round(diversidad, 3),
            f"{speaker}_frecuencia_palabras_accion": grupo["palabras_accion"].sum()
        }
        resumenes.append(resumen)

    final_dict = {}
    for r in resumenes:
        final_dict.update(r)

    return pd.DataFrame([final_dict])


