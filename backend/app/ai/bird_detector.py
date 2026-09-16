from xml.parsers.expat import model

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


# Load BirdNET model
analyzer = None


def load_birdnet():

    global analyzer

    if analyzer is not None:
        return analyzer

    try:

        print("🐦 Loading BirdNET model...")

        analyzer = Analyzer()

        print("✅ BirdNET model loaded")

        return analyzer

    except Exception as e:

        print(
            f"❌ BirdNET model could not be loaded: {e}"
        )

        analyzer = None

        return None


def detect_birds(audio_path: str):

    model = load_birdnet()
    if model is None:
        return []

    recording = Recording(
    model,
    audio_path,
    min_conf=0.1,
)

    recording.analyze()

    predictions = []

    for detection in recording.detections:

        predictions.append({
            "species": detection["common_name"],
            "scientific_name": detection["scientific_name"],
            "confidence": round(
                float(detection["confidence"]),
                4
            ),
        })

    return predictions