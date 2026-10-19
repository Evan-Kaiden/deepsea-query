from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

import preprocess
from embedder import Embedder
from store import DATASET_DIR, Store

LEGACY_MARKER = DATASET_DIR / ".embed_mode"  # mode of the old per-image .pt files
BATCH = 256


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_embedder():
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    return Embedder(
        model, preprocess.build(preprocess_val), tokenizer, device=pick_device()
    )



def main():
    store = Store()
    store.sync()
    todo = store.pending(preprocess.MODE)

    # vectors from the old .pt layout are reused if they match the current mode;
    # ones predating the marker came from open_clip's Resize+CenterCrop
    legacy_mode = LEGACY_MARKER.read_text().strip() if LEGACY_MARKER.exists() else "crop"
    reuse_legacy = legacy_mode == preprocess.MODE

    embedder = None
    done = reused = 0
    batch = []
    for path in todo:
        pt = Path(path).with_suffix(".pt")
        if reuse_legacy and pt.exists():
            vec = torch.load(pt, weights_only=False)
            reused += 1
        else:
            if embedder is None:  # skip the model load when nothing needs embedding
                embedder = load_embedder()
            vec = embedder.embed_image(Image.open(path))
            done += 1
        batch.append((path, F.normalize(vec.float().cpu(), dim=0)))
        if len(batch) == BATCH:
            store.put(batch, preprocess.MODE)
            batch = []
            print(f"{done + reused:6d} / {len(todo)}")
    store.put(batch, preprocess.MODE)

    print(f"\n{done} embedded, {reused} imported from .pt, mode={preprocess.MODE}")


if __name__ == "__main__":
    main()
