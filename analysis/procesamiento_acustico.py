def procesar_acustico(audio_path, segmentos_path, diarizacion_path):
    import librosa
    import pandas as pd
    import json
    from collections import Counter

    from analysis.volumen import calcular_volumen
    from analysis.velocidad import calcular_velocidad
    from analysis.pitch import analizar_pitch
    from analysis.emocion_audio import detectar_emocion_audio

    print("📥 Cargando audio...")
    y, sr = librosa.load(audio_path, sr=None)

    with open(segmentos_path, "r", encoding="utf-8") as f:
        segmentos = json.load(f)

    with open(diarizacion_path, "r", encoding="utf-8") as f:
        diarizacion = json.load(f)

    resultados = []
    for i, seg in enumerate(segmentos):
        start = seg['start']
        end = seg['end']
        speaker = seg['speaker']
        texto = seg.get('text', '').strip()

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segmento = y[start_sample:end_sample]

        volumen = calcular_volumen(y_segmento)
        velocidad = calcular_velocidad(texto, start, end)
        pitch_data = analizar_pitch(y_segmento, sr)
        emocion_audio = detectar_emocion_audio(audio_path, start, end)

        if i == 0:
            silencio_previo = 0.0
        else:
            silencio_previo = round(start - segmentos[i - 1]['end'], 2)
            silencio_previo = max(0.0, silencio_previo)

        es_silencio_largo = silencio_previo > 1.5

        resultados.append({
            'speaker': speaker,
            'duracion': round(end - start, 2),
            'volumen_rms': volumen,
            'velocidad_palabras_seg': velocidad,
            'pitch_mean_hz': pitch_data['pitch_mean'],
            'pitch_min_hz': pitch_data['pitch_min'],
            'pitch_max_hz': pitch_data['pitch_max'],
            'pitch_range_hz': pitch_data['pitch_max'] - pitch_data['pitch_min'] if pitch_data['pitch_max'] and pitch_data['pitch_min'] else None,
            'num_palabras': len(texto.split()),
            'tiempo_silencio_previo': silencio_previo,
            'es_silencio_largo': es_silencio_largo,
            'emocion_audio': emocion_audio
        })

    df = pd.DataFrame(resultados)

    # === Agregación por speaker ===
    df_agregado = df.groupby("speaker").agg({
        'duracion': 'sum',
        'volumen_rms': 'mean',
        'velocidad_palabras_seg': 'mean',
        'pitch_mean_hz': 'mean',
        'pitch_min_hz': 'min',
        'pitch_max_hz': 'max',
        'pitch_range_hz': 'mean',
        'num_palabras': 'sum',
        'tiempo_silencio_previo': 'mean',
        'es_silencio_largo': 'sum',
        'emocion_audio': lambda x: Counter(x).most_common(1)[0][0] if len(x) else None
    }).reset_index()

    # Renombrar columnas por speaker
    filas = []
    for _, row in df_agregado.iterrows():
        speaker = row['speaker']
        prefijo = "speaker0_" if speaker == "SPEAKER_00" else "speaker1_"
        fila = {f"{prefijo}{k}": v for k, v in row.drop("speaker").items()}
        filas.append(fila)

    # Consolidar en una sola fila
    fila_final = {}
    for d in filas:
        fila_final.update(d)

    return pd.DataFrame([fila_final])
