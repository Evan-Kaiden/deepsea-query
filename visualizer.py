"""Simple tkinter GUI to visualize a query image and click through retrieved samples."""

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageTk

import preprocess

NUM_RETURN = 5
IMG_SIZE = 550  # starting side of each image box, and the paint mask's resolution
MIN_SCALE = 0.6  # smallest window, as a fraction of the starting size
RESIZE_DEBOUNCE_MS = 80

# must mirror the vision tower the Embedder feeds: BioCLIP is ViT-B/16 @ 224
MODEL_RES = preprocess.MODEL_RES
PATCH = 16
GRID = MODEL_RES // PATCH  # 14
PAINT_EPS = 0.05  # floor so unpainted patches keep some weight (and log(w) is finite)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fit_pil(image, size=IMG_SIZE):
    image = image.convert("RGB").copy()
    image.thumbnail((size, size), Image.LANCZOS)
    return image


def fit(image, size=IMG_SIZE):
    return ImageTk.PhotoImage(fit_pil(image, size))


_KERNELS = {}


def brush_kernel(radius, softness):
    """Round stamp with a Gaussian falloff, as an 'L' image of size 2*radius+1."""
    key = (radius, round(softness, 2))
    if key in _KERNELS:
        return _KERNELS[key]
    d = np.linspace(-1.0, 1.0, 2 * radius + 1)
    yy, xx = np.meshgrid(d, d, indexing="ij")
    r = np.hypot(xx, yy)
    if softness <= 0.01:
        a = (r <= 1.0).astype(np.float32)
    else:
        a = np.exp(-(r ** 2) / (2.0 * softness ** 2))
        a[r > 1.0] = 0.0
        a /= a.max()
    _KERNELS[key] = Image.fromarray((a * 255.0).astype(np.uint8), mode="L")
    return _KERNELS[key]


def crop_box(size):
    """The part of the display image the model actually receives.

    Under preprocess.MODE == "pad" this is the whole frame; under "crop" it is
    the center square, and paint outside it is discarded before the patch grid
    exists.
    """
    return preprocess.content_box(size)


def paint_to_weights(paint, eps=PAINT_EPS):
    """Paint ('L', display size) -> (GRID*GRID,) weights, exactly what Embedder
    will turn into the log-bias on the CLS attention row.

    The mask goes through the same preprocess.to_model_frame() as the image, so
    a stroke lands on the patches covering what it was drawn over. The mask is
    kept at display rather than original resolution: the geometry is aspect-
    preserving either way, and it all collapses to 14x14 regardless."""
    framed = preprocess.to_model_frame(paint, Image.BILINEAR)
    p = torch.from_numpy(np.asarray(framed).astype(np.float32) / 255.0)
    coverage = torch.nn.functional.avg_pool2d(p[None, None], PATCH)[0, 0]
    w = eps + (1.0 - eps) * coverage
    return (w / w.max()).flatten()


class App:
    def __init__(self, root):
        self.root = root
        root.title("deepsea-query visualizer")

        self.querier = None
        self.results = []
        self.scores = []
        self.index = 0
        self.query_path = None
        self._photos = {}

        # painting state
        self.query_pil = None    # original-resolution query image
        self.display_pil = None  # query fit to IMG_SIZE — the paint mask's frame
        self.view_pil = None     # query fit to the current box — what's on the canvas
        self.paint = None        # 'L' mask at display_pil resolution — the source of truth
        self.last_pt = None
        self.view = IMG_SIZE     # current side of each square image box
        self._resize_job = None

        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")

        self.open_btn = ttk.Button(controls, text="Open image…", command=self.open_image)
        self.open_btn.pack(side="left")

        ttk.Label(controls, text="  text query:").pack(side="left")
        self.text_var = tk.StringVar()
        entry = ttk.Entry(controls, textvariable=self.text_var, width=30)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _e: self.run_query())

        ttk.Label(controls, text="  top-k:").pack(side="left")
        self.k_var = tk.StringVar(value=str(NUM_RETURN))
        ttk.Spinbox(controls, from_=1, to=50, width=4, textvariable=self.k_var).pack(side="left")

        self.search_btn = ttk.Button(controls, text="Search", command=self.run_query)
        self.search_btn.pack(side="left", padx=8)

        brush = ttk.Frame(root, padding=(8, 0, 8, 8))
        brush.pack(fill="x")

        ttk.Label(brush, text="brush:").pack(side="left")
        self.radius_var = tk.IntVar(value=28)
        ttk.Scale(brush, from_=4, to=90, variable=self.radius_var, length=110).pack(side="left", padx=4)

        ttk.Label(brush, text="  softness:").pack(side="left")
        self.softness_var = tk.DoubleVar(value=0.5)
        ttk.Scale(brush, from_=0.0, to=1.0, variable=self.softness_var, length=110).pack(side="left", padx=4)

        ttk.Label(brush, text="  strength:").pack(side="left")
        self.strength_var = tk.DoubleVar(value=1.0)
        ttk.Scale(brush, from_=0.05, to=1.0, variable=self.strength_var, length=110).pack(side="left", padx=4)

        self.erase_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(brush, text="erase", variable=self.erase_var).pack(side="left", padx=6)
        ttk.Button(brush, text="Clear", command=self.clear_paint).pack(side="left")
        ttk.Button(brush, text="Weights…", command=self.show_weights).pack(side="left", padx=6)

        panes = ttk.Frame(root, padding=8)
        panes.pack(fill="both", expand=True)

        left = ttk.LabelFrame(panes, text="Query — drag to paint importance", padding=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.query_canvas = tk.Canvas(
            left, width=IMG_SIZE, height=IMG_SIZE, highlightthickness=0, cursor="dot"
        )
        self.query_canvas.pack()
        self.query_canvas.bind("<Button-1>", self.on_paint_start)
        self.query_canvas.bind("<B1-Motion>", self.on_paint_move)
        self.query_canvas.bind("<ButtonRelease-1>", self.on_paint_end)
        self.query_label = ttk.Label(left, text="(no image)", wraplength=IMG_SIZE)
        self.query_label.pack(pady=4)
        self.paint_label = ttk.Label(left, text="paint: none", wraplength=IMG_SIZE)
        self.paint_label.pack()

        right = ttk.LabelFrame(panes, text="Retrieved", padding=8)
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.result_canvas = tk.Canvas(right, width=IMG_SIZE, height=IMG_SIZE, highlightthickness=0)
        self.result_canvas.pack()
        # three blank lines reserve the label's full height so the window's
        # measured shape already includes it
        self.result_label = ttk.Label(right, text="\n\n", wraplength=IMG_SIZE, justify="center")
        self.result_label.pack(pady=4)

        nav = ttk.Frame(right)
        nav.pack()
        self.prev_btn = ttk.Button(nav, text="◀ Prev", command=lambda: self.step(-1), state="disabled")
        self.prev_btn.pack(side="left", padx=4)
        self.counter = ttk.Label(nav, text="0 / 0", width=10, anchor="center")
        self.counter.pack(side="left")
        self.next_btn = ttk.Button(nav, text="Next ▶", command=lambda: self.step(1), state="disabled")
        self.next_btn.pack(side="left", padx=4)

        self.status = ttk.Label(root, text="Loading model…", padding=6, relief="sunken", anchor="w")
        # packed ahead of the panes so a tight window squeezes them, not the status bar
        self.status.pack(fill="x", side="bottom", before=panes)

        root.bind("<Left>", lambda _e: self.step(-1))
        root.bind("<Right>", lambda _e: self.step(1))

        # lock the window to the layout's natural shape; the image boxes scale with it
        root.update_idletasks()
        base_w, base_h = root.winfo_reqwidth(), root.winfo_reqheight()
        self.chrome = (base_w - 2 * IMG_SIZE, base_h - IMG_SIZE)  # space that isn't image
        root.geometry(f"{base_w}x{base_h}")  # stop the window tracking its widgets' sizes
        root.aspect(base_w, base_h, base_w, base_h)
        min_w = max(int(base_w * MIN_SCALE), controls.winfo_reqwidth(), brush.winfo_reqwidth())
        root.minsize(min_w, round(min_w * base_h / base_w))
        root.bind("<Configure>", self.on_resize)

        self.set_busy(True)
        threading.Thread(target=self.load_backend, daemon=True).start()

    def on_resize(self, event):
        # the toplevel's bindtag fires for every child too; only the window counts
        if event.widget is not self.root:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(RESIZE_DEBOUNCE_MS, self.apply_resize)

    def apply_resize(self):
        self._resize_job = None
        chrome_w, chrome_h = self.chrome
        view = max(1, min((self.root.winfo_width() - chrome_w) // 2,
                          self.root.winfo_height() - chrome_h))
        if view == self.view:
            return
        self.view = view
        for canvas in (self.query_canvas, self.result_canvas):
            canvas.config(width=view, height=view)
        for label in (self.query_label, self.paint_label, self.result_label):
            label.config(wraplength=view)
        if self.query_pil is not None:
            self.view_pil = fit_pil(self.query_pil, view)
            self.redraw_query()
        self.render()

    def load_backend(self):
        # imported here so the window paints before open_clip pulls in the model
        import open_clip

        from embedder import Embedder
        from query import Query
        from store import DB_PATH, Store

        if not DB_PATH.exists():
            self.root.after(0, lambda: self.status.config(
                text=f"No {DB_PATH} — run embed_dataset.py first"))
            return

        device = pick_device()
        model, _, preprocess_val = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
        # same transform the dataset was embedded with — see preprocess.py
        embedder = Embedder(model, preprocess.build(preprocess_val), tokenizer, device=device)
        store = Store(read_only=True)
        querier = Query(embedder, store)
        n, stale = store.counts(preprocess.MODE)
        note = f"  {stale} vectors use another mode — rerun embed_dataset.py" if stale else ""

        def done():
            self.querier = querier
            self.set_busy(False)
            self.status.config(text=f"Ready — {n} samples on {device}{note}")

        self.root.after(0, done)

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.open_btn.config(state=state)
        self.search_btn.config(state=state)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a query image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.query_path = path
        self.query_pil = Image.open(path).convert("RGB")
        self.display_pil = fit_pil(self.query_pil)
        self.view_pil = fit_pil(self.query_pil, self.view)
        self.paint = Image.new("L", self.display_pil.size, 0)
        self.query_label.config(text=os.path.basename(path))
        self.redraw_query()

    # ---- painting -------------------------------------------------------

    def view_offset(self):
        """Top-left of the query image inside its square box."""
        return (self.view - self.view_pil.width) // 2, (self.view - self.view_pil.height) // 2

    def view_to_paint(self):
        """Factor from on-screen pixels to paint-mask pixels."""
        return self.paint.width / self.view_pil.width

    def redraw_query(self):
        """Composite the paint over the query image as a warm tint."""
        if self.view_pil is None:
            return
        shown = self.view_pil
        if self.paint is not None and self.paint.getbbox() is not None:
            tint = Image.new("RGB", shown.size, (255, 90, 40))
            mask = self.paint.resize(shown.size, Image.BILINEAR)
            shown = Image.composite(Image.blend(shown, tint, 0.45), shown, mask)

        # under "crop", darken what preprocessing throws away so paint that can
        # never reach the model is visibly outside the live area; under "pad"
        # the box is the whole frame and nothing is drawn
        box = crop_box(shown.size)
        clipped = box != (0.0, 0.0, float(shown.width), float(shown.height))
        if clipped:
            outside = Image.new("L", shown.size, 255)
            ImageDraw.Draw(outside).rectangle(box, fill=0)
            shown = Image.composite(
                Image.blend(shown, Image.new("RGB", shown.size, (0, 0, 0)), 0.55),
                shown, outside,
            )

        ox, oy = self.view_offset()
        self._photos["query"] = ImageTk.PhotoImage(shown)
        self.query_canvas.delete("all")
        self.query_canvas.create_image(ox, oy, anchor="nw", image=self._photos["query"])
        if clipped:
            x0, y0, x1, y1 = box
            self.query_canvas.create_rectangle(
                x0 + ox, y0 + oy, x1 + ox, y1 + oy, outline="#38bdf8", dash=(4, 3)
            )
        self.update_paint_label()

    def brush_radius(self):
        """The brush slider is in screen pixels; the mask may be at another scale."""
        return self.radius_var.get() * self.view_to_paint()

    def stamp(self, x, y):
        """Stamp the brush at paint-mask coords (x, y)."""
        radius = max(1, int(round(self.brush_radius())))
        kernel = brush_kernel(radius, float(self.softness_var.get()))
        if not self.erase_var.get():
            strength = float(self.strength_var.get())
            kernel = kernel.point(lambda v: int(v * strength))

        cx, cy = int(round(x)), int(round(y))
        box = (cx - radius, cy - radius, cx + radius + 1, cy + radius + 1)
        region = self.paint.crop(box)  # crop pads with 0 outside the image
        # lighter() lets overlapping strokes saturate at `strength` instead of summing
        merged = ImageChops.darker(region, ImageChops.invert(kernel)) if self.erase_var.get() \
            else ImageChops.lighter(region, kernel)
        self.paint.paste(merged, box)

    def stroke(self, x, y):
        """Stamp along the segment from the last point so fast drags don't gap."""
        if self.last_pt is None:
            self.stamp(x, y)
        else:
            x0, y0 = self.last_pt
            steps = max(1, int(max(abs(x - x0), abs(y - y0)) / max(1.0, self.brush_radius() / 4)))
            for i in range(1, steps + 1):
                self.stamp(x0 + (x - x0) * i / steps, y0 + (y - y0) * i / steps)
        self.last_pt = (x, y)

    def event_to_paint(self, event):
        """Canvas event coords -> paint-mask coords."""
        ox, oy = self.view_offset()
        s = self.view_to_paint()
        return (event.x - ox) * s, (event.y - oy) * s

    def on_paint_start(self, event):
        if self.paint is None:
            return
        self.last_pt = None
        self.stroke(*self.event_to_paint(event))
        self.redraw_query()

    def on_paint_move(self, event):
        if self.paint is None:
            return
        self.stroke(*self.event_to_paint(event))
        self.redraw_query()

    def on_paint_end(self, _event):
        self.last_pt = None

    def clear_paint(self):
        if self.paint is None:
            return
        self.paint = Image.new("L", self.display_pil.size, 0)
        self.redraw_query()

    def paint_weights(self):
        """The (196,) weight vector Embedder would consume, or None if unpainted."""
        if self.paint is None or self.paint.getbbox() is None:
            return None
        return paint_to_weights(self.paint)

    def lost_paint_fraction(self):
        """Share of painted intensity that falls outside the model's crop."""
        if self.paint is None or self.paint.getbbox() is None:
            return 0.0
        total = np.asarray(self.paint, dtype=np.float32).sum()
        if total == 0:
            return 0.0
        kept = np.asarray(self.paint.crop(tuple(round(v) for v in crop_box(self.paint.size))),
                          dtype=np.float32).sum()
        return float(1.0 - kept / total)

    def update_paint_label(self):
        w = self.paint_weights()
        if w is None:
            self.paint_label.config(text="paint: none — query would run unweighted")
            return

        lost = self.lost_paint_fraction()
        if lost > 0.01:
            note = f"  ⚠ {lost:.0%} of paint is outside the crop (discarded)"
        else:
            note = ""
        # the floor sits at eps/max(w) rather than exactly eps, so compare to the
        # observed minimum instead of the constant
        above = int((w > w.min() + 1e-4).sum())
        self.paint_label.config(
            text=f"paint: {GRID}×{GRID} weights  min {w.min():.2f}  mean {w.mean():.2f}  "
                 f"max {w.max():.2f}  ({above}/{w.numel()} patches touched){note}"
        )

    def show_weights(self):
        w = self.paint_weights()
        if w is None:
            self.status.config(text="Nothing painted yet.")
            return
        print("patch weights", tuple(w.shape), "\n", w.reshape(GRID, GRID))

        top = tk.Toplevel(self.root)
        top.title(f"patch weights — {GRID}×{GRID}, eps={PAINT_EPS}")
        text = tk.Text(top, width=GRID * 6 + 2, height=GRID + 2, font=("Menlo", 10))
        text.pack(padx=8, pady=8)
        grid = w.reshape(GRID, GRID)
        for row in grid:
            text.insert("end", " ".join(f"{v:5.2f}" for v in row) + "\n")
        text.config(state="disabled")

    def run_query(self):
        if self.querier is None:
            return
        text = self.text_var.get().strip() or None
        if self.query_path is None and text is None:
            self.status.config(text="Pick an image or type a text query first.")
            return

        try:
            k = max(1, int(self.k_var.get()))
        except ValueError:
            k = NUM_RETURN

        image = Image.open(self.query_path) if self.query_path else None
        weights = self.paint_weights() if image is not None else None
        self.set_busy(True)
        self.status.config(text="Searching…")

        def work():
            ids, scores = self.querier.query(
                num_return=k, image=image, text=text, attn_mask=weights
            )
            self.root.after(0, lambda: self.show_results(ids, [float(s) for s in scores]))

        threading.Thread(target=work, daemon=True).start()

    def show_results(self, ids, scores):
        self.results = ids
        self.scores = scores
        self.index = 0
        self.set_busy(False)
        self.status.config(text=f"{len(ids)} results")
        self.render()

    def step(self, delta):
        if not self.results:
            return
        self.index = (self.index + delta) % len(self.results)
        self.render()

    def render(self):
        if not self.results:
            return
        path = self.results[self.index]
        self._photos["result"] = fit(Image.open(path), self.view)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(self.view // 2, self.view // 2, image=self._photos["result"])
        label = os.path.basename(os.path.dirname(path))
        self.result_label.config(
            text=f"{label}\n{os.path.basename(path)}\nsimilarity: {self.scores[self.index]:.4f}"
        )
        self.counter.config(text=f"{self.index + 1} / {len(self.results)}")
        state = "normal" if len(self.results) > 1 else "disabled"
        self.prev_btn.config(state=state)
        self.next_btn.config(state=state)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
