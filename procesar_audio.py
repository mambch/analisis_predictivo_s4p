import os
import json
import whisperx
import pandas as pd
from pyannote.audio import Pipeline
from huggingface_hub.utils import HfHubHTTPError

from analysis.procesamiento_acustico import procesar_acustico
from analysis.procesamiento_semantico import procesar_semantico

# === CONFIGURACIÓN ===
token = "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"
AUDIO_DIR = "audio"
OUTPUT_DIR = "salidas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

registros_finales = []

def procesar_audio(audio_path):
    print(f"\n🎧 Procesando: {audio_path}")
    base = os.path.splitext(os.path.basename(audio_path))[0]

    # === Transcripción con WhisperX ===
    model = whisperx.load_model("medium", device="cpu", compute_type="float32")
    result = model.transcribe(audio_path)

    if not result.get("segments"):
        print("❌ Sin segmentos en la transcripción.")
        return None

    full_text = result.get("text") or " ".join(seg.get("text", "") for seg in result["segments"])
    with open(os.path.join(OUTPUT_DIR, f"{base}_transcripcion_completa.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(os.path.join(OUTPUT_DIR, f"{base}_transcripcion_raw.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # === Diarización con PyAnnote ===
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
        diarization = pipeline(audio_path)
    except HfHubHTTPError as e:
        print("❌ Error cargando modelo de diarización:", e)
        return None

    segments = [{
        "start": round(turn.start, 2),
        "end": round(turn.end, 2),
        "speaker": speaker
    } for turn, _, speaker in diarization.itertracks(yield_label=True)]

    if not segments:
        print("⚠️ No se detectaron segmentos de voz.")
        return None

    diarizacion_path = os.path.join(OUTPUT_DIR, f"{base}_diarizacion.json")
    with open(diarizacion_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    # === Asociación de texto a hablantes ===
    print("🧩 Asociando segmentos...")
    MARGEN = 0.5
    aligned_segments = []
    for seg in segments:
        seg_text = ""
        for word in result.get("segments", []):
            mid = (word.get("start", 0) + word.get("end", 0)) / 2
            if seg["start"] - MARGEN <= mid <= seg["end"] + MARGEN:
                seg_text += word.get("text", "").strip() + " "
        aligned_segments.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg_text.strip()
        })

    transcrip_path = os.path.join(OUTPUT_DIR, f"{base}_transcripcion_por_segmento.json")
    with open(transcrip_path, "w", encoding="utf-8") as f:
        json.dump(aligned_segments, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, f"{base}_transcripcion_por_segmento.txt"), "w", encoding="utf-8") as f:
        for s in aligned_segments:
            f.write(f"[{s['speaker']} - {s['start']}s → {s['end']}s]: {s['text']}\n")

    # === Procesamiento Acústico y Semántico ===
    print("🎙️ Ejecutando análisis acústico...")
    df_audio = procesar_acustico(audio_path, transcrip_path, diarizacion_path)
    df_audio.to_csv(os.path.join(OUTPUT_DIR, f"{base}_resultados_analisis.csv"), index=False)

    print("🧠 Ejecutando análisis semántico...")
    df_sem = procesar_semantico(transcrip_path)
    df_sem.to_csv(os.path.join(OUTPUT_DIR, f"{base}_resultados_semantico.csv"), index=False)

    # === Consolidación final: UNA FILA por grabación ===
    df_final = pd.concat([df_audio, df_sem], axis=1)
    df_final["archivo"] = base

    # === Etiquetado automático según nombre del archivo
    if base.lower().startswith("venta"):
        df_final["venta"] = 1
    elif base.lower().startswith("no_venta"):
        df_final["venta"] = 0
    else:
        df_final["venta"] = None

    # === Porcentaje de habla por speaker
    dur_cols = [col for col in df_final.columns if col.endswith("_duracion_total")]
    if len(dur_cols) == 2:
        total = df_final[dur_cols[0]].values[0] + df_final[dur_cols[1]].values[0]
        for col in dur_cols:
            pct_col = col.replace("_duracion_total", "_pct_habla")
            df_final[pct_col] = (df_final[col] / total * 100).round(2)

    # === Guardar fila individual y devolver
    df_final.to_csv(os.path.join(OUTPUT_DIR, f"{base}_registro_modelo.csv"), index=False)
    print(f"✅ Finalizado: {audio_path}")
    return df_final

# === EJECUCIÓN GLOBAL ===
if __name__ == "__main__":
    archivos = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
    if not archivos:
        print("❌ No se encontraron archivos .wav en /audio/")

    for audio_file in archivos:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        fila = procesar_audio(audio_path)
        if fila is not None:
            registros_finales.append(fila)

    if registros_finales:
        df_dataset = pd.concat(registros_finales, ignore_index=True)
        df_dataset.to_csv(os.path.join(OUTPUT_DIR, "dataset_final.csv"), index=False)
        print("📦 Dataset consolidado guardado en: salidas/dataset_final.csv")
