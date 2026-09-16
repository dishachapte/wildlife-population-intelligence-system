# ============================================================
# LAZY-LOADED BIRDNET ANALYZER
# ============================================================

analyzer = None


def load_birdnet():
    """
    Load BirdNET only when bird detection is requested.

    This prevents BirdNET from consuming memory during
    FastAPI startup.
    """

    global analyzer

    # Already loaded
    if analyzer is not None:
        return analyzer

    try:

        print(
            "🐦 Loading BirdNET model..."
        )

        # IMPORTANT:
        # Heavy BirdNET imports happen only when needed.
        from birdnetlib.analyzer import Analyzer

        analyzer = Analyzer()

        print(
            "✅ BirdNET model loaded successfully"
        )

        return analyzer

    except Exception as e:

        print(
            f"❌ Failed to load BirdNET: {e}"
        )

        analyzer = None

        return None


# ============================================================
# BIRD DETECTION
# ============================================================

def detect_birds(
    audio_path: str
):

    # --------------------------------------------------------
    # Import Recording only when needed
    # --------------------------------------------------------

    from birdnetlib import Recording


    # --------------------------------------------------------
    # Load analyzer
    # --------------------------------------------------------

    birdnet_analyzer = load_birdnet()


    if birdnet_analyzer is None:

        raise RuntimeError(
            "BirdNET model is not available"
        )


    # --------------------------------------------------------
    # Create recording
    # --------------------------------------------------------

    recording = Recording(
        birdnet_analyzer,
        audio_path,
        min_conf=0.1,
    )


    # --------------------------------------------------------
    # Analyze recording
    # --------------------------------------------------------

    recording.analyze()


    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    predictions = []


    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for detection in recording.detections:

        predictions.append({

            "species":
                detection[
                    "common_name"
                ],

            "scientific_name":
                detection[
                    "scientific_name"
                ],

            "confidence":
                round(
                    float(
                        detection[
                            "confidence"
                        ]
                    ),
                    4
                ),

        })


    # --------------------------------------------------------
    # Return predictions
    # --------------------------------------------------------

    return predictions