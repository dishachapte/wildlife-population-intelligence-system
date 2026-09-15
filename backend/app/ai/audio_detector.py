import numpy as np
import librosa
import csv
import tensorflow as tf
import tensorflow_hub as hub


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
labels = []
model_loaded = False


# ============================================================
# LOAD YAMNET MODEL ONLY WHEN NEEDED
# ============================================================

def load_yamnet():

    global model
    global labels
    global model_loaded

    # Already loaded
    if model_loaded:
        return model is not None

    print("🔊 Loading YAMNet audio model...")

    try:

        model = hub.load(
            "https://tfhub.dev/google/yamnet/1"
        )

        print("✅ YAMNet audio model loaded successfully")

        # ====================================================
        # LOAD YAMNET CLASS LABELS
        # ====================================================

        class_map_path = (
            model
            .class_map_path()
            .numpy()
            .decode("utf-8")
        )

        with tf.io.gfile.GFile(class_map_path) as csv_file:

            reader = csv.DictReader(csv_file)

            labels = [
                row["display_name"]
                for row in reader
            ]

        print(
            f"✅ Loaded {len(labels)} audio labels"
        )

        model_loaded = True

        return True

    except Exception as e:

        print(
            f"⚠️ YAMNet audio model could not be loaded: {e}"
        )

        print(
            "⚠️ Audio recognition is temporarily unavailable."
        )

        model = None
        labels = []
        model_loaded = True

        return False


# ============================================================
# AUDIO DETECTION
# ============================================================

def detect_audio(audio_path: str):

    """
    Analyze an audio file using YAMNet.

    Returns the top 5 detected audio events.
    """

    # ========================================================
    # LOAD MODEL ONLY WHEN AUDIO IS ACTUALLY USED
    # ========================================================

    if not load_yamnet():

        return [
            {
                "label": "Audio model unavailable",
                "confidence": 0.0
            }
        ]

    # ========================================================
    # CHECK LABELS
    # ========================================================

    if not labels:

        return [
            {
                "label": "Audio labels unavailable",
                "confidence": 0.0
            }
        ]

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    try:

        waveform, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

    except Exception as e:

        print(
            f"❌ Failed to load audio: {e}"
        )

        return [
            {
                "label": "Audio loading failed",
                "confidence": 0.0
            }
        ]

    # ========================================================
    # CHECK AUDIO
    # ========================================================

    if waveform is None or len(waveform) == 0:

        return [
            {
                "label": "No audio detected",
                "confidence": 0.0
            }
        ]

    # ========================================================
    # RUN YAMNET
    # ========================================================

    try:

        scores, embeddings, spectrogram = model(
            waveform
        )

    except Exception as e:

        print(
            f"❌ YAMNet prediction failed: {e}"
        )

        return [
            {
                "label": "Audio analysis failed",
                "confidence": 0.0
            }
        ]

    # ========================================================
    # AVERAGE PREDICTION SCORES
    # ========================================================

    mean_scores = (
        tf.reduce_mean(
            scores,
            axis=0
        )
        .numpy()
    )

    # ========================================================
    # GET TOP 5 PREDICTIONS
    # ========================================================

    top5 = (
        np.argsort(
            mean_scores
        )[-5:][::-1]
    )

    predictions = []

    # ========================================================
    # CREATE RESULTS
    # ========================================================

    for index in top5:

        index = int(index)

        if index >= len(labels):
            continue

        predictions.append({

            "label":
                labels[index],

            "confidence":
                round(
                    float(
                        mean_scores[index]
                    ),
                    4
                )

        })

    return predictions