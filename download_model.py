import os
import re
import json
import logging
import glob
import time
from huggingface_hub import snapshot_download
from utils import timer_decorator

logging.basicConfig(level=logging.INFO)

BASE_DIR = "/"
TOKENIZER_PATTERNS = [["*.json", "tokenizer*"]]
MODEL_PATTERNS = [["*.safetensors"], ["*.bin"], ["*.pt"]]
CODE_PATTERNS = [["*.py"]] if os.getenv("TRUST_REMOTE_CODE") and os.getenv("TRUST_REMOTE_CODE").lower() == "true" else None


@timer_decorator
def download(name, revision, type, cache_dir):
    if type == "model":
        pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
        if CODE_PATTERNS:
            pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] + CODE_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
        else:
            pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
    elif type == "tokenizer":
        pattern_sets = TOKENIZER_PATTERNS
    else:
        raise ValueError(f"Invalid type: {type}")

    try:
        for pattern_set in pattern_sets:
            try:
                logging.info(f"starting download of mode {name} with pattern {pattern_set}")
                path = snapshot_download(name, revision=revision, cache_dir=cache_dir, allow_patterns=pattern_set)
                for pattern in pattern_set:
                    if glob.glob(os.path.join(path, pattern)):
                        logging.info(f"Successfully downloaded {pattern} model files.")
                return path
            except RuntimeError as e:
                logging.info("Une erreur de runtime est survenue.")
                if "HF_HUB_ENABLE_HF_TRANSFER" in str(e) and os.getenv("HF_HUB_ENABLE_HF_TRANSFER"):
                    logging.warning("Échec avec hf_transfer, désactivation pour les prochains téléchargements")
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                    # Réessayer immédiatement avec hf_transfer désactivé
                    path = snapshot_download(name, revision=revision, cache_dir=cache_dir, allow_patterns=pattern_set)
                    return path
                else:
                    raise
    except ValueError:
        raise ValueError(f"No patterns matching {pattern_sets} found for download.")
    except Exception as e:
        logging.error(f"Erreur FATALE : {e}")


def __call__():
    cache_dir = os.getenv("HF_HOME")
    model_name, model_revision = os.getenv("MODEL_NAME"), os.getenv("MODEL_REVISION") or None
    logging.info(f"démarrage du téléchargement du modèle: {model_name}")
    model_path = download(model_name, model_revision, "model", cache_dir)
    metadata = {
        "MODEL_NAME": model_path,
        "MODEL_REVISION": model_revision,
    }

    for k, v in metadata.items():
        if v not in (None, ""):
            os.setenv(k, v)

    # with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
    #     json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)


if __name__ == "__main__":
    __call__()
