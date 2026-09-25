from pathlib import Path

import open_clip
import torch
from PIL import Image

import preprocess
from embedder import Embedder

DATASET_DIR = Path("dataset")
MODE_MARKER = DATASET_DIR / ".embed_mode"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    if MODE_MARKER.exists():
        previous = MODE_MARKER.read_text().strip()
    else:
        # vectors predating the marker came from open_clip's Resize+CenterCrop
        previous = "crop" if any(DATASET_DIR.glob("*/*.pt")) else None

    stale = previous is not None and previous != preprocess.MODE
    if stale:
        print(
            f"preprocessing changed ({previous} -> {preprocess.MODE}); "
            "recomputing every vector"
        )

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    embedder = Embedder(
        model, preprocess.build(preprocess_val), tokenizer, device=pick_device()
    )

    done = skipped = 0
    for class_dir in sorted(d for d in DATASET_DIR.iterdir() if d.is_dir()):
        for img_path in sorted(class_dir.glob("*.png")):
            vec_path = img_path.with_suffix(".pt")
            if vec_path.exists() and not stale:
                skipped += 1
                continue

            enc = embedder.embed_image(Image.open(img_path))
            torch.save(enc.cpu(), vec_path)
            done += 1

        print(f"{class_dir.name:32s} {done:6d} embedded  {skipped:6d} cached")

    MODE_MARKER.write_text(preprocess.MODE + "\n")
    print(f"\n{done} embedded, {skipped} reused, mode={preprocess.MODE}")


if __name__ == "__main__":
    main()
