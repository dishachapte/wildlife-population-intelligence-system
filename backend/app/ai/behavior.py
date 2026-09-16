from pathlib import Path
from pyexpat import model

from ultralytics import YOLO


# ============================================================
# BEHAVIOR MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BEHAVIOR_MODEL_PATH = (
    BASE_DIR / "behavior_model.pt"
)


# ============================================================
# LOAD TRAINED BEHAVIOR MODEL
# ============================================================

behavior_model = None


def load_behavior_model():

    global behavior_model

    if behavior_model is not None:
        return behavior_model

    try:
        print("🧠 Loading behavior model...")

        behavior_model = YOLO(
            str(BEHAVIOR_MODEL_PATH)
        )

        print(
            f"✅ Behavior model loaded: "
            f"{BEHAVIOR_MODEL_PATH}"
        )

        return behavior_model

    except Exception as e:

        print(
            f"❌ Failed to load behavior model: {e}"
        )

        behavior_model = None

        return None


# ============================================================
# BEHAVIOR CLASSES
# ============================================================

# These MUST match the classes in your Roboflow dataset.
#
# data.yaml:
#
# names:
#   0: DRINKING
#   1: Standing
#   2: eating
#   3: lying
#
# We normalize them to clean display names.

BEHAVIOR_CLASSES = {

    "DRINKING": "Drinking",

    "Standing": "Standing",

    "eating": "Eating",

    "lying": "Lying",

}


# ============================================================
# NORMALIZE BEHAVIOR NAME
# ============================================================

def normalize_behavior(
    behavior_name: str
) -> str:

    if not behavior_name:

        return "Unknown"


    behavior_name = str(
        behavior_name
    ).strip()


    return BEHAVIOR_CLASSES.get(

        behavior_name,

        behavior_name.title()

    )


# ============================================================
# DETECT BEHAVIOR FROM ANIMAL CROP
# ============================================================

def detect_behavior(image):
    model = load_behavior_model()

    if model is None:

        print("❌ Behavior model is not loaded")

        return {
            "behavior": "Unknown",
            "confidence": 0.0
        }

    try:

        results = model.predict(
            source=image,
            verbose=False,
            conf=0.05
        )

    except Exception as e:

        print(
            f"❌ Behavior model error: {e}"
        )

        return {
            "behavior": "Unknown",
            "confidence": 0.0
        }

    if not results:

        print(
            "⚠️ Behavior model returned no results"
        )

        return {
            "behavior": "Unknown",
            "confidence": 0.0
        }

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:

        print(
            "⚠️ No behavior detected "
            "even at confidence 0.05"
        )

        return {
            "behavior": "standing",
            "confidence": 0.0
        }

    print(
        f"🔎 Behavior model found "
        f"{len(result.boxes)} behavior boxes"
    )

    best_index = 0
    best_confidence = 0.0

    for index, box in enumerate(result.boxes):

        confidence = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )

        class_name = behavior_model.names[
            class_id
        ]

        print(
            f"   Behavior candidate: "
            f"{class_name} "
            f"confidence={confidence:.2f}"
        )

        if confidence > best_confidence:

            best_confidence = confidence

            best_index = index

    best_box = result.boxes[
        best_index
    ]

    class_id = int(
        best_box.cls[0]
    )

    behavior_name = behavior_model.names[
        class_id
    ]

    behavior_name = normalize_behavior(
        behavior_name
    )

    print(
        f"✅ Selected behavior: "
        f"{behavior_name} "
        f"confidence={best_confidence:.2f}"
    )

    return {

        "behavior":
            behavior_name,

        "confidence":
            round(
                best_confidence,
                2
            )

    }