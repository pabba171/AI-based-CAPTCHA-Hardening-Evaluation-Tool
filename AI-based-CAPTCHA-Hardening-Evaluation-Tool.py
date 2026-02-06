# =========================================
# AI-based CAPTCHA Hardening & Evaluation Tool 
# =========================================

# 0) Install deps
!pip -q install captcha tensorflow scikit-learn

# 1) Imports
import numpy as np, random, string, io
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from tabulate import tabulate
import numpy as np


# 2) Config
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_W, IMG_H = 160, 60
TEXT_LEN = 5
CLASSES = list(string.ascii_uppercase + string.digits) # 36 classes
N_CLASSES = len(CLASSES)
BATCH_SIZE = 32
EPOCHS = 5

# For a faster demo you can drop to 2000; increase to improve accuracy.
NUM_SAMPLES_PER_TYPE = 2000

# 3) Utils
char_to_idx = {c:i for i,c in enumerate(CLASSES)}
idx_to_char = {i:c for c,i in char_to_idx.items()}

def random_text(length=TEXT_LEN):
  return ''.join(random.choices(CLASSES, k=length))

def to_onehot_first_char(text_list):
  y_idx = np.array([char_to_idx[t[0]] for t in text_list], dtype=np.int32)
  y = tf.keras.utils.to_categorical(y_idx, num_classes=N_CLASSES)
  return y

# 4) CAPTCHA generators (variants)
def gen_clean(text):
  return ImageCaptcha(width=IMG_W, height=IMG_H).generate_image(text)

def gen_noisy(text):
  img = gen_clean(text).convert("RGB")
  noise = np.random.randint(0, 40, img.size[::-1] + (3,), dtype=np.uint8)
  arr = np.clip(np.array(img) + noise, 0, 255).astype(np.uint8)
  return Image.fromarray(arr)

def gen_lines(text, n_lines=8):
  img = gen_clean(text).convert("RGB")
  draw = ImageDraw.Draw(img)
  for _ in range(n_lines):
    x1, y1 = np.random.randint(0, IMG_W), np.random.randint(0, IMG_H)
    x2, y2 = np.random.randint(0, IMG_W), np.random.randint(0, IMG_H)
  draw.line((x1, y1, x2, y2), fill=(np.random.randint(256),)*3, width=1)
  return img

def gen_blur(text):
  return gen_clean(text).filter(ImageFilter.GaussianBlur(radius=1.2))

def gen_warp(text):
  img = gen_clean(text).convert("RGB")
  w, h = img.size
  # Define original corners
  orig_corners = [(0, 0), (w, 0), (w, h), (0, h)]
  # Jitter destination corners
  jitter = 10
  new_corners = [
        (np.random.randint(-jitter, jitter), np.random.randint(-jitter, jitter)),
        (w + np.random.randint(-jitter, jitter), np.random.randint(-jitter, jitter)),
        (w + np.random.randint(-jitter, jitter), h + np.random.randint(-jitter, jitter)),
        (np.random.randint(-jitter, jitter), h + np.random.randint(-jitter, jitter)),
  ]
  coeffs = _find_perspective_coeffs(orig_corners, new_corners)
  return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

def _find_perspective_coeffs(pa, pb):
  """Find coefficients for perspective transform between two point sets."""
  matrix = []
  for p1, p2 in zip(pa, pb):
    matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
    matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
  A = np.array(matrix)
  B = np.array([p[0] for p in pb] + [p[1] for p in pb])
  res = np.linalg.lstsq(A, B, rcond=None)[0]
  return res

def gen_inverted(text):
  return ImageOps.invert(gen_clean(text).convert("RGB"))

VARIANTS = {
  "clean": gen_clean,
  "noisy": gen_noisy,
  "lines": gen_lines,
  "blur": gen_blur,
  "warp": gen_warp,
  "invert": gen_inverted,
}

# 5) Build datasets per variant
def build_variant_dataset(variant_name, n=NUM_SAMPLES_PER_TYPE):
  X, y_txt = [], []
  fn = VARIANTS[variant_name]
  for _ in range(n):
    txt = random_text(TEXT_LEN)
    img = fn(txt)
    arr = np.array(img, dtype=np.float32) / 255.0
    X.append(arr); y_txt.append(txt)
  X = np.stack(X)
  y = to_onehot_first_char(y_txt)
  return X, y, y_txt

datasets = {}
for v in VARIANTS.keys():
  X, y, y_txt = build_variant_dataset(v)
  datasets[v] = {"X": X, "y": y, "txt": y_txt}

# 6) Train/Val split (attacker trained on clean only)
X_train, X_val, y_train, y_val = train_test_split(
  datasets["clean"]["X"], datasets["clean"]["y"], test_size=0.2, random_state=SEED, stratify=np.argmax(datasets["clean"]["y"], axis=1)
)

# 7) Transfer-learning solver (EfficientNetB0 head → predict 1st char)
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, 3))
base.trainable = False # freeze backbone for speed/stability

inp = base.input
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.25)(x)
x = Dense(256, activation="relu")(x)
out = Dense(N_CLASSES, activation="softmax")(x)
model = Model(inp, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
  X_train, y_train,
  validation_data=(X_val, y_val),
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  verbose=1
)

# 8) Evaluate crackability per variant
def eval_variant(name):
  X, y = datasets[name]["X"], datasets[name]["y"]
  loss, acc = model.evaluate(X, y, verbose=0)
  return acc

acc_per_variant = {name: eval_variant(name) for name in VARIANTS.keys()}
acc_clean = acc_per_variant["clean"]
eps = 1e-6
security_score = {name: max(0.0, 1.0 - (acc_per_variant[name] / max(acc_clean, eps))) for name in VARIANTS.keys()}

print("\n=== Accuracy per variant (crackability) ===")
for k,v in acc_per_variant.items():
  print(f"{k:>7}: {v:.4f}")

print("\n=== Security score (higher = harder to crack) ===")
for k,v in security_score.items():
  print(f"{k:>7}: {v:.4f}")

# 9) Dashboard: training curve + crackability + security
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training Progress")
plt.legend()

plt.subplot(1,3,2)
plt.bar(list(acc_per_variant.keys()), list(acc_per_variant.values()))
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Crackability (accuracy) by CAPTCHA type")

plt.subplot(1,3,3)
plt.bar(list(security_score.keys()), list(security_score.values()))
plt.xticks(rotation=30)
plt.ylabel("Security Score (1 - acc_var/acc_clean)")
plt.title("Security by CAPTCHA type (higher is better)")
plt.tight_layout()
plt.show()

# 10) Confusion matrix on the hardest variant (lowest accuracy)
hardest = min(acc_per_variant, key=lambda k: acc_per_variant[k])
preds = model.predict(datasets[hardest]["X"], verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(datasets[hardest]["y"], axis=1)
cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
plt.figure(figsize=(10,10))
disp.plot(cmap="Blues", values_format="d", xticks_rotation=90)
plt.title(f"Confusion Matrix — hardest variant: {hardest}")
plt.show()

# 11) Visual examples (Pred vs True) for each variant
def show_examples_for_variant(name, n=10):
  X = datasets[name]["X"][:n]
  txt = datasets[name]["txt"][:n]
  preds = model.predict(X, verbose=0)
  pred_c = [idx_to_char[i] for i in np.argmax(preds, axis=1)]
  true_c = [t[0] for t in txt]
  cols = 5
  rows = int(np.ceil(n/cols))
  plt.figure(figsize=(3*cols, 2.3*rows))
  for i in range(n):
    plt.subplot(rows, cols, i+1)
    plt.imshow(X[i])
    ok = (pred_c[i]==true_c[i])
    plt.title(f"P:{pred_c[i]}  T:{true_c[i]}", color=("green" if ok else "red"))
    plt.axis("off")
  plt.suptitle(f"Variant: {name}  (examples)", y=1.02, fontsize=12)
  plt.tight_layout()
  plt.show()

for v in VARIANTS.keys():
  show_examples_for_variant(v, n=10)

# 12) Estimating full 5-letter CAPTCHA accuracy & security score

# Variants
variants = ["clean", "noisy", "lines", "blur", "warp", "invert"]

# 1-letter accuracies (from your results)
one_letter_acc = np.array([acc_per_variant[v] for v in variants])

# Estimate full 5-letter CAPTCHA accuracy
five_letter_acc = one_letter_acc ** 5

# Estimate security score (higher = harder)
security_scores = 1 - one_letter_acc

# Create dataframe
df = pd.DataFrame({
  "Variant": variants,
  "Estimated 5-letter Accuracy": five_letter_acc,
  "Security Score": security_scores
})

# Display nicely in console
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))