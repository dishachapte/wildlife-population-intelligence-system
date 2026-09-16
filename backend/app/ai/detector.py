from pathlib import Path

import cv2
from ultralytics import YOLO

from app.ai.behavior import detect_behavior


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent.parent

SPECIES_MODEL_PATH = BACKEND_DIR / "yolov8n.pt"


# ============================================================
# LOAD SPECIES MODEL
# ============================================================

model = None


def load_species_model():

    global model

    if model is not None:
        return model

    try:

        print("🦁 Loading species YOLO model...")

        model = YOLO(
            str(SPECIES_MODEL_PATH)
        )

        print(
            f"✅ Species model loaded: "
            f"{SPECIES_MODEL_PATH}"
        )

        return model

    except Exception as e:

        print(
            f"❌ Failed to load species model: {e}"
        )

        model = None

        return None


# ============================================================
# CONSERVATION STATUS
# ============================================================

CONSERVATION_STATUS = {

    "elephant": "Endangered",

    "lion": "Vulnerable",

    "tiger": "Endangered",

    "deer": "Least Concern",

    "zebra": "Near Threatened",

    "giraffe": "Vulnerable",

    "bird": "Protected",

    "bear": "Vulnerable",

    "dog": "Least Concern",

    "cat": "Least Concern",

    "cow": "Least Concern",

    "horse": "Least Concern",

    "sheep": "Least Concern",

    "goat": "Least Concern",

    "monkey": "Least Concern",

}


# ============================================================
# GET CONSERVATION STATUS
# ============================================================

def get_conservation_status(
    species: str
):

    species = (
        species
        .lower()
        .strip()
    )

    return CONSERVATION_STATUS.get(

        species,

        "Not Evaluated"

    )


# ============================================================
# DETECT OBJECTS + BEHAVIOR
# ============================================================

def detect_objects(
    image_path: str
):

    """
    Detects multiple animals in an image.

    For each detected animal:

    1. Detect species using YOLOv8.
    2. Get its bounding box.
    3. Expand the bounding box slightly.
    4. Crop the individual animal with extra context.
    5. Send the crop to the trained behavior model.
    6. Return species, behavior, confidence,
       conservation status, and bounding box.

    Multiple animals are processed individually.
    """

    # ========================================================
    # START ANALYSIS
    # ========================================================

    print(
        f"🔍 Starting AI analysis: "
        f"{image_path}"
    )


    # ========================================================
    # CHECK SPECIES MODEL
    # ========================================================

    model = load_species_model()
    if model is None:
        raise RuntimeError("Species YOLO model could not be loaded.")

    # ========================================================
    # CHECK IMAGE EXISTS
    # ========================================================

    image_file = Path(
        image_path
    )


    if not image_file.exists():

        raise FileNotFoundError(

            f"Image file does not exist: "
            f"{image_path}"

        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        str(image_file)
    )


    if image is None:

        raise ValueError(

            f"OpenCV could not read image: "
            f"{image_path}"

        )


    print(

        f"✅ Image loaded: "
        f"{image.shape}"

    )


    # ========================================================
    # RUN SPECIES DETECTION
    # ========================================================

    print(
        "🦁 Running species detection..."
    )


    results = model.predict(

        source=image,

        verbose=False,

        conf=0.10,

    )


    # ========================================================
    # STORE ALL DETECTIONS
    # ========================================================

    detections = []


    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    height, width = (
        image.shape[:2]
    )


    # ========================================================
    # PROCESS EACH DETECTION RESULT
    # ========================================================

    for result in results:

        # ----------------------------------------------------
        # CHECK IF BOXES EXIST
        # ----------------------------------------------------

        if result.boxes is None:

            continue


        print(

            f"📦 Detected boxes: "
            f"{len(result.boxes)}"

        )


        # ----------------------------------------------------
        # PROCESS EVERY DETECTED ANIMAL
        # ----------------------------------------------------

        for box in result.boxes:

            # =================================================
            # SPECIES CLASS
            # =================================================

            cls = int(
                box.cls[0]
            )


            # =================================================
            # SPECIES CONFIDENCE
            # =================================================

            confidence = float(
                box.conf[0]
            )


            # =================================================
            # SPECIES NAME
            # =================================================

            species = model.names[
                cls
            ]


            species = (

                species
                .lower()
                .strip()

            )


            print(

                f"🐘 Animal detected: "
                f"{species} "
                f"({confidence:.2f})"

            )


            # =================================================
            # GET BOUNDING BOX
            # =================================================

            x1, y1, x2, y2 = (

                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)

            )


            # =================================================
            # CLAMP ORIGINAL COORDINATES
            # =================================================

            x1 = max(

                0,

                min(
                    x1,
                    width - 1
                )

            )


            y1 = max(

                0,

                min(
                    y1,
                    height - 1
                )

            )


            x2 = max(

                0,

                min(
                    x2,
                    width
                )

            )


            y2 = max(

                0,

                min(
                    y2,
                    height
                )

            )


            # =================================================
            # DEFAULT BEHAVIOR RESULT
            # =================================================

            behavior_result = {

                "behavior":
                    "Unknown",

                "confidence":
                    0.0

            }


            # =================================================
            # CHECK VALID BOUNDING BOX
            # =================================================

            if (

                x2 > x1

                and

                y2 > y1

            ):

                # =============================================
                # CALCULATE ANIMAL BOX SIZE
                # =============================================

                box_width = (
                    x2 - x1
                )

                box_height = (
                    y2 - y1
                )


                # =============================================
                # ADD 25% PADDING
                # =============================================

                padding_x = int(

                    box_width
                    * 0.50

                )


                padding_y = int(

                    box_height
                    * 0.50

                )


                # =============================================
                # EXPANDED CROP COORDINATES
                # =============================================

                crop_x1 = max(

                    0,

                    x1 - padding_x

                )


                crop_y1 = max(

                    0,

                    y1 - padding_y

                )


                crop_x2 = min(

                    width,

                    x2 + padding_x

                )


                crop_y2 = min(

                    height,

                    y2 + padding_y

                )


                # =============================================
                # CREATE ANIMAL CROP
                # =============================================

                animal_crop = image[

                    crop_y1:crop_y2,

                    crop_x1:crop_x2

                ]


                # =============================================
                # CHECK CROP
                # =============================================

                if (

                    animal_crop is not None

                    and

                    animal_crop.size > 0

                ):

                    print(

                        f"🧠 Running behavior "
                        f"analysis for {species}..."

                    )


                    # =========================================
                    # RUN TRAINED BEHAVIOR MODEL
                    # =========================================
                    print(
                        f"🖼️ Behavior crop for {species}: "
                        f"shape={animal_crop.shape}"
                    )
                    cv2.imwrite(
                        f"debug_{species}_{x1}_{y1}.jpg",
                        animal_crop
                    )
                    behavior_result = (

                        detect_behavior(

                            animal_crop

                        )

                    )


                    print(

                        f"➡️ Behavior: "
                        f"{behavior_result['behavior']} "
                        f"("
                        f"{behavior_result['confidence']:.2f}"
                        f")"

                    )


            # =================================================
            # GET CONSERVATION STATUS
            # =================================================

            conservation_status = (

                get_conservation_status(

                    species

                )

            )


            # =================================================
            # ADD COMPLETE DETECTION
            # =================================================

            detections.append({

                "species":
                    species,

                "confidence":
                    round(

                        confidence,

                        2

                    ),

                "behavior":
                    behavior_result[

                        "behavior"

                    ],

                "behavior_confidence":
                    behavior_result[

                        "confidence"

                    ],

                "conservation_status":
                    conservation_status,

                "bounding_box": {

                    "x1":
                        int(x1),

                    "y1":
                        int(y1),

                    "x2":
                        int(x2),

                    "y2":
                        int(y2),

                },

            })


    # ========================================================
    # FINAL ANALYSIS RESULT
    # ========================================================

    print(

        f"✅ AI analysis complete. "
        f"Animals detected: "
        f"{len(detections)}"

    )


    # ========================================================
    # RETURN ALL DETECTIONS
    # ========================================================

    return detections