"""
interface.py — Book Rating Predictor · MLP from Scratch
Features:
  - Hyperparameter tuning: LR, Epochs, Hidden Layers (GUI)
  - Training loss curve (Matplotlib popup)
  - Error tolerance test (feature masking)
  - Stratified random book draw
  - Age feature from Users.csv
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mlp_scratch import ScratchMLP
from preprocessing import load_and_clean_data, get_train_data

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0D0D1A",
    "surface":  "#13132A",
    "surface2": "#1C1C3A",
    "border":   "#2A2A5A",
    "primary":  "#7C6FFF",
    "danger":   "#FF4D6D",
    "success":  "#00D68F",
    "warning":  "#FFB830",
    "text":     "#EAEAF5",
    "text2":    "#7B7B9D",
    "entry":    "#0A0A18",
    "tag_bg":   "#1E1E40",
}
F = {
    "display": ("Georgia",  20, "bold"),
    "sub":     ("Arial",     8),
    "section": ("Arial",    10, "bold"),
    "label":   ("Arial",     9),
    "book":    ("Georgia",  11, "italic"),
    "score":   ("Arial",    13, "bold"),
    "verdict": ("Arial",    11, "bold"),
    "btn":     ("Arial",    10, "bold"),
    "btn_lg":  ("Arial",    11, "bold"),
    "mono":    ("Courier",   9),
}

FEATURES = ['user_mean', 'book_mean', 'user_count_norm', 'book_count_norm', 'age_norm']
FEATURE_LABELS = ['User Avg', 'Book Avg', 'User Activity', 'Book Popularity', 'User Age']

# ── Helpers ───────────────────────────────────────────────────────────────────
def _lighten(h, a=25):
    h = h.lstrip("#")
    r,g,b = (int(h[i:i+2],16) for i in (0,2,4))
    return "#{:02x}{:02x}{:02x}".format(min(255,r+a),min(255,g+a),min(255,b+a))

def _darken(h, a=30):
    h = h.lstrip("#")
    r,g,b = (int(h[i:i+2],16) for i in (0,2,4))
    return "#{:02x}{:02x}{:02x}".format(max(0,r-a),max(0,g-a),max(0,b-a))

def make_button(parent, text, command, bg, fg, font, pady=9, pack_kwargs=None):
    lbl = tk.Label(parent, text=text, bg=bg, fg=fg, font=font,
                   cursor="hand2", pady=pady, padx=10)
    pk = pack_kwargs or {"fill": "x"}
    lbl.pack(**pk)
    def on_enter(e):   lbl.config(bg=_lighten(bg))
    def on_leave(e):   lbl.config(bg=bg)
    def on_press(e):   lbl.config(bg=_darken(bg));  command()
    def on_release(e): lbl.config(bg=_lighten(bg))
    lbl.bind("<Enter>",           on_enter)
    lbl.bind("<Leave>",           on_leave)
    lbl.bind("<ButtonPress-1>",   on_press)
    lbl.bind("<ButtonRelease-1>", on_release)
    return lbl

def _divider(parent, color="#2A2A5A"):
    tk.Frame(parent, height=1, bg=color).pack(fill="x", pady=8)

def stretch(raw, raw_min, raw_max, gamma=0.7):
    span = (raw_max - raw_min) or 1.0
    p    = float(np.clip((raw - raw_min) / span, 0.0, 1.0))
    c    = p - 0.5
    mag  = (abs(c) ** gamma) * (2 ** (1 - gamma)) / 2
    p_s  = 0.5 + np.sign(c) * mag
    return float(np.clip(p_s * 9.0 + 1.0 + np.random.uniform(-0.2, 0.2), 1.0, 10.0))


# ── App ────────────────────────────────────────────────────────────────────────
class BookApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Rating Predictor · MLP from Scratch")
        self.root.geometry("620x980")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)

        self.model        = None
        self.full_df      = None
        self.current_book = None
        self.raw_min      = 0.20
        self.raw_max      = 0.70
        self.loss_history = []   # (epoch, mse) tuples for plot

        # Feature mask checkboxes (error tolerance)
        self.feat_mask = [tk.BooleanVar(value=True) for _ in FEATURES]

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        self._header()
        self._train_panel()
        self._predict_panel()
        self._tolerance_panel()
        self._result_panel()

    def _header(self):
        hdr = tk.Frame(self.root, bg=C["bg"])
        hdr.pack(fill="x", pady=(24, 4))
        tk.Label(hdr, text="📖  Book Rating Predictor",
                 font=F["display"], bg=C["bg"], fg=C["text"]).pack()
        tk.Label(hdr,
                 text="Book-Crossing Dataset  ·  Multilayer Perceptron  ·  NumPy only",
                 font=F["sub"], bg=C["bg"], fg=C["text2"]).pack(pady=(3,0))
        tk.Frame(self.root, height=1, bg=C["primary"]).pack(fill="x", padx=30, pady=(8,0))

    def _train_panel(self):
        outer = tk.Frame(self.root, bg=C["surface"], padx=20, pady=16)
        outer.pack(padx=24, fill="x", pady=(14,0))

        tk.Label(outer, text="⚙️  Model Training", font=F["section"],
                 bg=C["surface"], fg=C["primary"]).pack(anchor="w")
        _divider(outer, C["border"])

        # ── Hyperparameters row ──
        params = tk.Frame(outer, bg=C["surface"])
        params.pack(fill="x", pady=(0, 10))

        def _param(parent, label, col):
            box = tk.Frame(parent, bg=C["surface"])
            box.grid(row=0, column=col, padx=(0,20), sticky="w")
            tk.Label(box, text=label, font=F["label"],
                     bg=C["surface"], fg=C["text2"]).pack(anchor="w")
            return box

        params.columnconfigure(3, weight=1)

        # LR
        lr_box = _param(params, "Learning Rate", 0)
        self.lr_spin = ttk.Spinbox(lr_box, from_=0.0001, to=0.5, increment=0.001, width=9)
        self.lr_spin.set(0.005); self.lr_spin.pack()

        # Epochs
        ep_box = _param(params, "Max Epochs", 1)
        self.epoch_spin = ttk.Spinbox(ep_box, from_=50, to=3000, increment=50, width=9)
        self.epoch_spin.set(1000); self.epoch_spin.pack()

        # Hidden Layers — NEW: user can pick topology
        hl_box = _param(params, "Hidden Layers", 2)
        self.layer_var = tk.StringVar(value="32,16")
        tk.Label(hl_box, text="sizes (comma-sep):", font=("Arial",8),
                 bg=C["surface"], fg=C["text2"]).pack(anchor="w")
        tk.Entry(hl_box, textvariable=self.layer_var, width=10,
                 bg=C["entry"], fg=C["text"], insertbackground=C["text"],
                 relief="flat", bd=4).pack()

        # Architecture preview (right)
        self.arch_lbl = tk.Label(params, text="5 → 32 → 16 → 1",
                                  font=F["mono"], bg=C["tag_bg"], fg=C["primary"],
                                  padx=8, pady=4)
        self.arch_lbl.grid(row=0, column=3, sticky="e")
        self.layer_var.trace_add("write", self._update_arch_label)

        # Buttons row
        btn_row = tk.Frame(outer, bg=C["surface"])
        btn_row.pack(fill="x", pady=(4,0))

        make_button(btn_row, "TRAIN MODEL  🚀", self._train_model,
                    bg=C["primary"], fg=C["text"], font=F["btn_lg"], pady=9,
                    pack_kwargs={"side":"left","fill":"x","expand":True,"padx":(0,6)})
        make_button(btn_row, "SHOW LOSS CURVE  📈", self._show_loss_curve,
                    bg=C["surface2"], fg=C["text"], font=F["btn"], pady=9,
                    pack_kwargs={"side":"left","fill":"x","expand":True})

        self.status_lbl = tk.Label(outer, text="Status: Ready — train to begin",
                                    font=F["label"], bg=C["surface"], fg=C["text2"])
        self.status_lbl.pack(pady=(8,0))

    def _predict_panel(self):
        outer = tk.Frame(self.root, bg=C["surface"], padx=20, pady=16)
        outer.pack(padx=24, fill="x", pady=(12,0))

        tk.Label(outer, text="🎯  Challenge the AI", font=F["section"],
                 bg=C["surface"], fg=C["danger"]).pack(anchor="w")
        _divider(outer, C["border"])

        make_button(outer, "DRAW A RANDOM BOOK  🎲", self._get_random_book,
                    bg=C["surface2"], fg=C["text"], font=F["btn"], pady=7)
        tk.Frame(outer, height=8, bg=C["surface"]).pack()

        # Book card
        card = tk.Frame(outer, bg=C["surface2"], padx=14, pady=10)
        card.pack(fill="x", pady=(0,12))
        self.book_title_lbl = tk.Label(card,
            text="Train the model, then draw a book.",
            font=F["book"], bg=C["surface2"], fg=C["text"],
            wraplength=480, justify="left")
        self.book_title_lbl.pack(anchor="w")
        self.book_author_lbl = tk.Label(card, text="",
            font=("Arial",9), bg=C["surface2"], fg=C["text2"])
        self.book_author_lbl.pack(anchor="w", pady=(4,0))
        self.book_meta_lbl = tk.Label(card, text="",
            font=("Arial",8), bg=C["surface2"], fg=C["text2"])
        self.book_meta_lbl.pack(anchor="w", pady=(2,0))

        # Guess
        guess_row = tk.Frame(outer, bg=C["surface"])
        guess_row.pack(fill="x")
        tk.Label(guess_row, text="Your Rating Guess  (1 – 10):",
                 font=F["label"], bg=C["surface"], fg=C["text2"]).pack(side="left")
        self.guess_entry = tk.Entry(guess_row,
            font=("Arial",16,"bold"), justify="center",
            bg=C["entry"], fg=C["text"], insertbackground=C["text"],
            relief="flat", bd=6, width=5)
        self.guess_entry.pack(side="right")

        tk.Frame(outer, height=12, bg=C["surface"]).pack()
        make_button(outer, "REVEAL RESULT  ✨", self._show_result,
                    bg=C["danger"], fg=C["text"], font=F["btn_lg"], pady=9)

    def _tolerance_panel(self):
        outer = tk.Frame(self.root, bg=C["surface"], padx=20, pady=14)
        outer.pack(padx=24, fill="x", pady=(12,0))

        tk.Label(outer, text="🔬  Error Tolerance — Feature Masking",
                 font=F["section"], bg=C["surface"], fg=C["warning"]).pack(anchor="w")
        tk.Label(outer,
                 text="Uncheck features to simulate missing input data and observe prediction robustness.",
                 font=("Arial",8), bg=C["surface"], fg=C["text2"]).pack(anchor="w", pady=(2,6))
        _divider(outer, C["border"])

        cb_row = tk.Frame(outer, bg=C["surface"])
        cb_row.pack(fill="x")
        for i, (feat_lbl, var) in enumerate(zip(FEATURE_LABELS, self.feat_mask)):
            cb = tk.Checkbutton(cb_row, text=feat_lbl, variable=var,
                                bg=C["surface"], fg=C["text"],
                                selectcolor=C["surface2"],
                                activebackground=C["surface"],
                                activeforeground=C["text"],
                                font=("Arial",9))
            cb.grid(row=0, column=i, padx=8, sticky="w")

        self.tolerance_lbl = tk.Label(outer, text="",
            font=("Arial",9,"italic"), bg=C["surface"], fg=C["warning"])
        self.tolerance_lbl.pack(pady=(6,0))

    def _result_panel(self):
        outer = tk.Frame(self.root, bg=C["bg"], padx=24, pady=10)
        outer.pack(padx=24, fill="x", pady=(10,0))
        self.scores_lbl = tk.Label(outer, text="", font=F["score"],
                                    bg=C["bg"], fg=C["text"],
                                    justify="center", wraplength=560)
        self.scores_lbl.pack()
        self.verdict_lbl = tk.Label(outer, text="", font=F["verdict"],
                                     bg=C["bg"], fg=C["text2"], justify="center")
        self.verdict_lbl.pack(pady=(4,0))

    # ── Arch label updater ────────────────────────────────────────────────────
    def _update_arch_label(self, *_):
        try:
            hidden = [int(x.strip()) for x in self.layer_var.get().split(",") if x.strip()]
            layers = [5] + hidden + [1]
            self.arch_lbl.config(text=" → ".join(str(l) for l in layers))
        except Exception:
            self.arch_lbl.config(text="invalid")

    def _parse_layers(self):
        hidden = [int(x.strip()) for x in self.layer_var.get().split(",") if x.strip()]
        return [5] + hidden + [1]   # 5 input features now

    # ── Training ──────────────────────────────────────────────────────────────
    def _train_model(self):
        try:
            layers = self._parse_layers()
        except Exception:
            messagebox.showerror("Invalid Architecture",
                                  "Enter hidden layer sizes as comma-separated integers, e.g. 32,16")
            return
        try:
            self._status("Loading data...  📚", C["text2"])
            self.root.update()

            self.full_df = load_and_clean_data()
            if self.full_df is None:
                messagebox.showerror("Error",
                    "Could not read CSV files.\nMake sure Books.csv, Ratings.csv, Users.csv are in the project folder.")
                return

            self._status("Sampling training set...", C["text2"])
            self.root.update()

            X, y = get_train_data(self.full_df)
            lr     = float(self.lr_spin.get())
            epochs = int(self.epoch_spin.get())

            self.model       = ScratchMLP(layers=layers, lr=lr, clip_value=1.0)
            self.loss_history = []

            best_loss    = float("inf")
            best_weights = None
            best_biases  = None
            patience     = 0
            MAX_PATIENCE = 6

            for epoch in range(epochs):
                out = self.model.forward(X)

                if np.any(np.isnan(out)):
                    self._status("⚠️  NaN — lower the learning rate!", C["danger"])
                    messagebox.showwarning("Overflow",
                        f"NaN at epoch {epoch}. Try a lower LR (e.g. 0.003).")
                    self.model = None
                    return

                self.model.backward(X, y, out)

                if epoch % 50 == 0:
                    loss = float(np.mean((y - out) ** 2))
                    self.loss_history.append((epoch, loss))
                    print(f"Epoch {epoch:>4}/{epochs}  |  MSE: {loss:.4f}")
                    self._status(f"Epoch {epoch}/{epochs}  ·  MSE: {loss:.4f}", C["text2"])
                    self.root.update()

                    if loss < best_loss - 0.001:
                        best_loss    = loss
                        best_weights = [w.copy() for w in self.model.weights]
                        best_biases  = [b.copy() for b in self.model.biases]
                        patience     = 0
                    else:
                        patience += 1
                        if patience >= MAX_PATIENCE:
                            print(f"Early stopping at epoch {epoch}  (best MSE: {best_loss:.4f})")
                            break

            # Restore best checkpoint
            if best_weights:
                self.model.weights = best_weights
                self.model.biases  = best_biases

            # Stretch bounds
            final_out  = self.model.forward(X)
            final_loss = float(np.mean((y - final_out) ** 2))
            self.raw_min = float(final_out.min()) - 0.05
            self.raw_max = float(final_out.max()) + 0.05

            print(f"\n{'='*46}")
            print(f"Training complete  |  Best MSE: {best_loss:.4f}  |  Final: {final_loss:.4f}")
            print(f"Raw  →  min: {final_out.min():.4f}  max: {final_out.max():.4f}  mean: {final_out.mean():.4f}")
            print(f"{'='*46}\n")

            self._status(f"Ready  ·  Best MSE: {best_loss:.4f}  ✅", C["success"])
            messagebox.showinfo("Training Complete",
                f"Model trained!\nBest MSE: {best_loss:.4f}\n\nDraw a book and challenge the AI.")

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self._status("Error ❌", C["danger"])

    # ── Loss curve ────────────────────────────────────────────────────────────
    def _show_loss_curve(self):
        if not self.loss_history:
            messagebox.showwarning("No Data", "Train the model first.")
            return

        epochs = [l[0] for l in self.loss_history]
        losses = [l[1] for l in self.loss_history]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0D0D1A")
        ax.set_facecolor("#13132A")
        ax.plot(epochs, losses, color="#7C6FFF", linewidth=2.5, label="Training MSE")
        ax.fill_between(epochs, losses, alpha=0.15, color="#7C6FFF")
        ax.scatter([epochs[np.argmin(losses)]], [min(losses)],
                   color="#00D68F", s=80, zorder=5,
                   label=f"Best MSE: {min(losses):.4f}")

        ax.set_title("Training Loss Curve — MLP from Scratch",
                     color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="#A0A8B8")
        ax.set_ylabel("MSE", color="#A0A8B8")
        ax.tick_params(colors="#A0A8B8")
        ax.legend(facecolor="#1C1C3A", labelcolor="white", edgecolor="#2A2A5A")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2A5A")

        plt.tight_layout()
        plt.show()

    # ── Random book ───────────────────────────────────────────────────────────
    def _get_random_book(self):
        if self.full_df is None:
            messagebox.showwarning("Not Trained", "Train the model first.")
            return

        pool = self.full_df[self.full_df["Rating"] <= 5] if random.random() < 0.35 \
               else self.full_df
        if len(pool) == 0:
            pool = self.full_df

        row = pool.sample(1).iloc[0]
        self.current_book = row

        self.book_title_lbl.config(text=f'"{row.get("Title","?")}"')
        self.book_author_lbl.config(text=f'— {row.get("Author","?")}')

        age = row.get("Age", None)
        age_str = f"User age: {int(age)}" if pd.notna(age) and age > 0 else "User age: unknown"
        self.book_meta_lbl.config(text=age_str)

        self.scores_lbl.config(text="")
        self.verdict_lbl.config(text="")
        self.tolerance_lbl.config(text="")
        self.guess_entry.delete(0, tk.END)

    # ── Show result ───────────────────────────────────────────────────────────
    def _show_result(self):
        if self.model is None:
            messagebox.showwarning("Not Trained", "Train the model first.")
            return
        if self.current_book is None:
            messagebox.showwarning("No Book", "Draw a book first.")
            return

        try:
            guess = float(self.guess_entry.get())
            if not (1 <= guess <= 10):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid", "Enter a number between 1 and 10.")
            return

        # Build feature vector — respect mask
        feat_vals = [self.current_book[f] for f in FEATURES]
        mask      = [var.get() for var in self.feat_mask]
        masked    = [v if m else 0.0 for v, m in zip(feat_vals, mask)]

        # Check which features are masked for display
        masked_names = [FEATURE_LABELS[i] for i, m in enumerate(mask) if not m]

        X_test = np.array([masked])
        raw    = self.model.forward(X_test)[0][0]

        if np.isnan(raw):
            messagebox.showerror("NaN", "Model output invalid. Retrain.")
            return

        ai   = stretch(raw, self.raw_min, self.raw_max)
        real = float(self.current_book["Rating"])

        self.scores_lbl.config(
            text=f"You: {guess:.1f} ⭐    AI: {ai:.1f} ⭐    Actual: {real:.0f} ⭐",
            fg=C["text"]
        )

        u_err = abs(guess - real)
        a_err = abs(ai    - real)

        if u_err < a_err - 0.05:
            v, col = f"🏆  You win!   (your error ±{u_err:.1f}  vs  AI ±{a_err:.1f})", C["success"]
        elif a_err < u_err - 0.05:
            v, col = f"🤖  AI wins!   (AI error ±{a_err:.1f}  vs  yours ±{u_err:.1f})", C["warning"]
        else:
            v, col = f"🤝  Tie!   (both ≈ ±{(u_err+a_err)/2:.1f})", C["text2"]

        self.verdict_lbl.config(text=v, fg=col)

        # Error tolerance message
        if masked_names:
            self.tolerance_lbl.config(
                text=f"⚠️  Masked features: {', '.join(masked_names)}  — "
                     f"AI predicted {ai:.1f} without this information.",
                fg=C["warning"]
            )
        else:
            self.tolerance_lbl.config(text="✅  All features active (full information)", fg=C["text2"])

    def _status(self, text, color):
        self.status_lbl.config(text=f"Status: {text}", fg=color)


import pandas as pd   # needed for pd.notna in _get_random_book

if __name__ == "__main__":
    root = tk.Tk()
    BookApp(root)
    root.mainloop()