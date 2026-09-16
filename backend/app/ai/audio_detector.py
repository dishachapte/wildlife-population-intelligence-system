import csv

import numpy as np


# ============================================================
# LAZY-LOADED YAMNET MODEL
# ============================================================

model = None

labels = []


# ============================================================
# LOAD YAMNET
# ============================================================

def load_yamnet():
    """
    Load TensorFlow + TensorFlow Hub + YAMNet only when
    audio analysis is actually requested.

    This prevents TensorFlow from consuming large amounts
    of memory during FastAPI startup.
    """

    global model
    global labels

    # Already loaded
    if model is not None:

        return model


    try:

        print(
            "🔊 Loading YAMNet audio model..."
        )


        # IMPORTANT:
        # These imports MUST stay inside this function.
        import tensorflow as tf
        import tensorflow_hub as hub


        # ----------------------------------------------------
        # Load YAMNet
        # ----------------------------------------------------

        model = hub.load(
            "https://tfhub.dev/google/yamnet/1"
        )


        print(
            "✅ YAMNet audio model loaded successfully"
        )


        # ----------------------------------------------------
        # Load class labels
        # ----------------------------------------------------

        class_map_path = (
            model
            .class_map_path()
            .numpy()
            .decode("utf-8")
        )


        with tf.io.gfile.GFile(
            class_map_path
        ) as csv_file:

            reader = csv.DictReader(
                csv_file
            )

            labels = [
                row["display_name"]
                for row in reader
            ]


        print(
            f"✅ Loaded "
            f"{len(labels)} audio labels"
        )


        return model


    except Exception as e:

        print(
            f"❌ Failed to load YAMNet: {e}"
        )

        model = None

        labels = []

        return None


# ============================================================
# AUDIO DETECTION
# ============================================================

def detect_audio(
    audio_path: str
):

    """
    Analyze an audio file using YAMNet.

    Returns the top 5 detected audio classes.
    """


    # --------------------------------------------------------
    # Load YAMNet
    # --------------------------------------------------------

    yamnet_model = load_yamnet()


    if (
        yamnet_model is None
        or not labels
    ):

        raise RuntimeError(
            "YAMNet model is not available"
        )


    # --------------------------------------------------------
    # Import heavy audio libraries only when needed
    # --------------------------------------------------------

    import librosa
    import tensorflow as tf


    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    try:

        waveform, sample_rate = (
            librosa.load(
                audio_path,
                sr=16000,
                mono=True
            )
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not load audio file: {e}"
        )


    # --------------------------------------------------------
    # Convert waveform
    # --------------------------------------------------------

    waveform = waveform.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Run YAMNet
    # --------------------------------------------------------

    try:

        scores, embeddings, spectrogram = (
            yamnet_model(
                waveform
            )
        )

    except Exception as e:

        raise RuntimeError(
            f"YAMNet analysis failed: {e}"
        )


    # --------------------------------------------------------
    # Average scores across time
    # --------------------------------------------------------

    mean_scores = (
        tf.reduce_mean(
            scores,
            axis=0
        )
    )


    # --------------------------------------------------------
    # Get top 5 classes
    # --------------------------------------------------------

    top5 = tf.argsort(
        mean_scores,
        direction="DESCENDING"
    )[:5]


    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    results = []


    for index in top5:

        index = int(
            index.numpy()
        )


        results.append({

            "label":
                labels[index],

            "confidence":
                round(
                    float(
                        mean_scores[
                            index
                        ].numpy()
                    ),
                    4
                )

        })


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return results