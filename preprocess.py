from PIL import Image

MODE = "pad"          # "pad" (full frame) or "crop" (open_clip default)
MODEL_RES = 224
PAD_FILL = (0, 0, 0)


def pad_to_square(img, fill=PAD_FILL):
    """Center the frame on a square canvas of side max(w, h).

    Keeps the source mode, so the visualizer's single-channel paint mask pads
    with 0 (no weight in the bars) through the very same call the image takes.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    if img.mode != "RGB":
        fill = 0
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def content_box(size, mode=MODE):
    """The part of an image at `size` that survives preprocessing, in its own
    coords. Full frame under "pad"; the center square under "crop"."""
    w, h = size
    if mode == "pad":
        return (0.0, 0.0, float(w), float(h))

    scale = MODEL_RES / min(w, h)
    rw, rh = max(MODEL_RES, round(w * scale)), max(MODEL_RES, round(h * scale))
    left, top = (rw - MODEL_RES) // 2, (rh - MODEL_RES) // 2
    return (left / scale, top / scale,
            (left + MODEL_RES) / scale, (top + MODEL_RES) / scale)


def to_model_frame(img, resample=Image.BICUBIC, mode=MODE, fill=PAD_FILL):
    """Replay the spatial half of preprocessing on any single-channel or RGB
    image: what comes back is MODEL_RES x MODEL_RES, aligned with the patch grid.

    The visualizer runs the paint mask through this so the brush and the model
    agree on where things are.
    """
    if mode == "pad":
        return pad_to_square(img, fill).resize((MODEL_RES, MODEL_RES), resample)

    w, h = img.size
    scale = MODEL_RES / min(w, h)
    img = img.resize(
        (max(MODEL_RES, round(w * scale)), max(MODEL_RES, round(h * scale))), resample
    )
    w, h = img.size
    left, top = (w - MODEL_RES) // 2, (h - MODEL_RES) // 2
    return img.crop((left, top, left + MODEL_RES, top + MODEL_RES))


def build(preprocess_val, mode=MODE):
    """Wrap open_clip's transform, swapping its Resize+CenterCrop for `mode`.

    The tail (mode conversion, to-tensor, Normalize) is reused as-is so the
    channel statistics still match what BioCLIP was trained on.
    """
    if mode == "crop":
        return preprocess_val

    from torchvision.transforms import Compose, Resize, CenterCrop

    tail = Compose([
        t for t in preprocess_val.transforms
        if not isinstance(t, (Resize, CenterCrop))
    ])

    def preprocess(img):
        return tail(to_model_frame(img.convert("RGB"), mode=mode))

    return preprocess
