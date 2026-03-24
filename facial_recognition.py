"""
Sistema de Reconocimiento Facial — MTCNN + ORB + MySQL
Captura múltiple → Entrenamiento → Reconocimiento en vivo
"""

import os
import cv2
import pickle
import numpy as np
from tkinter import *
from tkinter import messagebox as msg
from tkinter import ttk
import database as db

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
DATASET_PATH    = "dataset/"
MODEL_PATH      = "model/orb_model.pkl"
TEMP_PATH       = "temp/"
IMG_SIZE        = (150, 200)
DEFAULT_PHOTOS  = 30
ORB_FEATURES    = 500
MATCH_THRESHOLD = 70
SIMILARITY_MIN  = 0.55

BG     = "#151515"
HEADER = "#101010"
BTN    = "#202020"
WHITE  = "#f4f5f4"
GREEN  = "#27ae60"
RED    = "#e74c3c"
YELLOW = "#f39c12"
FONT   = "Century Gothic"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("model",      exist_ok=True)
os.makedirs(TEMP_PATH,    exist_ok=True)

_detector = None
def get_detector():
    global _detector
    if _detector is None:
        from mtcnn.mtcnn import MTCNN
        _detector = MTCNN()
    return _detector

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# ──────────────────────────────────────────────
#  UTILIDADES DE ROSTRO
# ──────────────────────────────────────────────

def detect_and_crop(frame_bgr):
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = get_detector().detect_faces(rgb)
    if not faces:
        return None
    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    crop = frame_bgr[y:y+h, x:x+w]
    return cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)

def compute_descriptors(img_gray):
    _, des = orb.detectAndCompute(img_gray, None)
    return des

def similarity(des1, des2):
    if des1 is None or des2 is None:
        return 0.0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < MATCH_THRESHOLD]
    return len(good) / len(matches)

# ──────────────────────────────────────────────
#  MODELO
# ──────────────────────────────────────────────

def train_model():
    model   = {}
    persons = [p for p in os.listdir(DATASET_PATH)
               if os.path.isdir(os.path.join(DATASET_PATH, p))]
    if not persons:
        return False, "No hay imagenes en el dataset. Registra personas primero."

    for person in persons:
        folder = os.path.join(DATASET_PATH, person)
        imgs   = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".png"))]
        descriptors = []
        for img_name in imgs:
            gray = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            des = compute_descriptors(gray)
            if des is not None:
                descriptors.append(des)
        if descriptors:
            model[person] = descriptors

    if not model:
        return False, "No se pudieron extraer descriptores."

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    total = sum(len(v) for v in model.values())
    return True, f"Modelo entrenado: {len(model)} persona(s), {total} imagen(es)."

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(frame_bgr, model):
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = get_detector().detect_faces(rgb)
    results = []

    for face_data in faces:
        x, y, w, h = face_data["box"]
        x, y = max(0, x), max(0, y)
        crop  = frame_bgr[y:y+h, x:x+w]
        crop  = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        des_q = compute_descriptors(gray)

        best_name, best_score = "Desconocido", 0.0
        for person, des_list in model.items():
            scores = [similarity(des_q, d) for d in des_list]
            avg    = float(np.mean(scores)) if scores else 0.0
            if avg > best_score:
                best_score, best_name = avg, person

        if best_score < SIMILARITY_MIN:
            best_name = "Desconocido"

        color = (0, 200, 80) if best_name != "Desconocido" else (60, 60, 220)
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame_bgr, f"{best_name} ({best_score:.0%})",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        results.append((best_name, best_score))

    return results

# ──────────────────────────────────────────────
#  GUI HELPERS
# ──────────────────────────────────────────────

def styled_btn(parent, text, cmd, color=BTN, w=32):
    return Button(parent, text=text, fg=WHITE, bg=color,
                  activebackground=BG, activeforeground=WHITE,
                  borderwidth=0, relief=FLAT,
                  font=(FONT, 13), height=2, width=w, command=cmd)

def make_header(parent, text):
    Label(parent, text=text, fg=WHITE, bg=HEADER,
          font=(FONT, 16), width=500, height=2).pack(fill=X)

def spacer(parent, n=1):
    for _ in range(n):
        Label(parent, text="", bg=BG).pack()

def status_lbl(parent):
    lbl = Label(parent, text="", fg=YELLOW, bg=BG,
                font=(FONT, 11), wraplength=480, justify=CENTER)
    lbl.pack()
    return lbl

def check_db_status(label):
    ok, info = db.testConnection()
    label.config(text=f"{'ok' if ok else 'err'}  BD: {info}",
                 fg=GREEN if ok else RED)

# ──────────────────────────────────────────────
#  PANTALLA: CAPTURA + REGISTRO EN BD
# ──────────────────────────────────────────────

def open_capture_screen():
    win = Toplevel(root)
    win.title("Registrar persona")
    win.geometry("540x450")
    win.configure(bg=BG)
    win.resizable(False, False)

    make_header(win, "Registrar persona")
    spacer(win)

    Label(win, text="Nombre del estudiante:", fg=WHITE, bg=BG, font=(FONT, 12)).pack()
    name_var = StringVar()
    Entry(win, textvariable=name_var, justify=CENTER, font=(FONT, 12), width=30).pack(ipady=5)
    spacer(win)

    Label(win, text="Numero de fotos a capturar:", fg=WHITE, bg=BG, font=(FONT, 12)).pack()
    count_var = IntVar(value=DEFAULT_PHOTOS)
    Spinbox(win, from_=5, to=200, textvariable=count_var,
            justify=CENTER, font=(FONT, 12), width=7).pack(ipady=4)
    spacer(win)

    progress = ttk.Progressbar(win, length=430, mode="determinate")
    progress.pack()
    spacer(win)
    status = status_lbl(win)

    def save_reference_to_db(name, folder):
        imgs = sorted(os.listdir(folder))
        if not imgs:
            return
        ref_path = os.path.join(folder, imgs[0])
        res = db.registerUser(name, ref_path)
        if res["affected"]:
            print(f"[DB] '{name}' guardado en BD (id={res['id']})")
        else:
            print(f"[DB] No se pudo guardar '{name}' en BD")

    def start_capture():
        name  = name_var.get().strip()
        total = count_var.get()

        if not name:
            status.config(text="Escribe el nombre del estudiante.", fg=RED)
            return

        folder = os.path.join(DATASET_PATH, name)
        os.makedirs(folder, exist_ok=True)

        # ── Contar fotos ya existentes para continuar la numeración ──────────
        existing = [f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".png"))]
        existing_count = len(existing)

        # Informar al usuario si ya tiene fotos
        if existing_count > 0:
            status.config(
                text=f"'{name}' ya tiene {existing_count} foto(s).\n"
                     f"Se agregarán {total} foto(s) nuevas (total: {existing_count + total}).",
                fg=YELLOW)
            win.update()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status.config(text="No se pudo abrir la camara.", fg=RED)
            return

        status.config(
            text=f"Capturando {total} fotos nuevas de '{name}'.\n"
                 f"Ya existentes: {existing_count}  |  ESPACIO = capturar  |  ESC = cancelar",
            fg=YELLOW)
        win.update()

        captured  = 0                     # fotos nuevas en esta sesión
        next_idx  = existing_count + 1    # índice para no sobreescribir
        progress["maximum"] = total
        progress["value"]   = 0

        while captured < total:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces   = get_detector().detect_faces(rgb)
            for f in faces:
                x, y, w, h = f["box"]
                cv2.rectangle(display, (x, y), (x+w, y+h), (40, 200, 80), 2)

            # Contador: nuevas en sesión / total sesión  |  total acumulado
            cv2.putText(display,
                        f"Nuevas: {captured}/{total}  |  Total acumulado: {existing_count + captured}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 200, 80), 2)
            cv2.putText(display, "ESPACIO=capturar  ESC=salir",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Captura de rostro", display)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == 32:
                crop = detect_and_crop(frame)
                if crop is not None:
                    # Nombre con índice acumulado: no sobreescribe fotos previas
                    img_path = os.path.join(folder, f"{name}_{next_idx:04d}.jpg")
                    cv2.imwrite(img_path, crop)
                    captured += 1
                    next_idx += 1
                    progress["value"] = captured
                    win.update()
                else:
                    warn = display.copy()
                    cv2.putText(warn, "SIN ROSTRO DETECTADO",
                                (50, display.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 60, 220), 2)
                    cv2.imshow("Captura de rostro", warn)
                    cv2.waitKey(500)

        cap.release()
        cv2.destroyAllWindows()

        if captured > 0:
            save_reference_to_db(name, folder)
            total_acum = existing_count + captured
            status.config(
                text=f"{captured} foto(s) nuevas agregadas para '{name}'.\n"
                     f"Total acumulado en dataset: {total_acum} foto(s).\n"
                     "Vuelve a entrenar el modelo para aplicar los cambios.",
                fg=GREEN)
        else:
            status.config(text="Captura cancelada sin fotos guardadas.", fg=YELLOW)

    spacer(win)
    styled_btn(win, "Iniciar captura", start_capture, color="#1a6b38").pack()

# ──────────────────────────────────────────────
#  PANTALLA: ENTRENAR MODELO
# ──────────────────────────────────────────────

def open_train_screen():
    win = Toplevel(root)
    win.title("Entrenar modelo")
    win.geometry("540x330")
    win.configure(bg=BG)
    win.resizable(False, False)

    make_header(win, "Entrenar modelo ORB")
    spacer(win, 2)

    persons    = [p for p in os.listdir(DATASET_PATH)
                  if os.path.isdir(os.path.join(DATASET_PATH, p))]
    total_imgs = sum(
        len(os.listdir(os.path.join(DATASET_PATH, p))) for p in persons
    )
    Label(win,
          text=f"Dataset local: {len(persons)} persona(s)  |  {total_imgs} imagen(es)",
          fg=YELLOW, bg=BG, font=(FONT, 12)).pack()

    db_lbl = Label(win, text="Verificando BD...", fg=YELLOW, bg=BG, font=(FONT, 11))
    db_lbl.pack()
    win.after(300, lambda: check_db_status(db_lbl))

    spacer(win)
    status = status_lbl(win)
    prog   = ttk.Progressbar(win, length=430, mode="indeterminate")
    prog.pack()
    spacer(win)

    def do_train():
        if not persons:
            status.config(text="No hay imagenes en el dataset.", fg=RED)
            return
        status.config(text="Entrenando... por favor espera.", fg=YELLOW)
        prog.start(10)
        win.update()
        ok, text = train_model()
        prog.stop()
        status.config(text=("OK  " if ok else "ERR  ") + text,
                      fg=GREEN if ok else RED)

    styled_btn(win, "Entrenar ahora", do_train, color="#7d4b00").pack()

# ──────────────────────────────────────────────
#  PANTALLA: RECONOCIMIENTO EN VIVO
# ──────────────────────────────────────────────

def open_recognition_screen():
    model = load_model()
    if model is None:
        msg.showwarning("Sin modelo",
                        "No se encontro un modelo entrenado.\n"
                        "Registra personas y entrena el modelo primero.")
        return

    win = Toplevel(root)
    win.title("Reconocimiento en vivo")
    win.geometry("540x260")
    win.configure(bg=BG)
    win.resizable(False, False)

    make_header(win, "Reconocimiento en vivo")
    spacer(win)

    Label(win,
          text=f"Modelo cargado: {len(model)} persona(s)\n"
               "Verde = reconocido  |  Rojo = desconocido\n"
               "Presiona ESC para cerrar la camara.",
          fg=WHITE, bg=BG, font=(FONT, 11), justify=CENTER).pack()
    spacer(win)
    status = status_lbl(win)

    def run_recognition():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status.config(text="No se pudo abrir la camara.", fg=RED)
            return
        status.config(text="Camara activa. ESC para detener.", fg=GREEN)
        win.update()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            predict(frame, model)
            cv2.imshow("Reconocimiento Facial  (ESC=salir)", frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        status.config(text="Reconocimiento detenido.", fg=YELLOW)

    styled_btn(win, "Iniciar reconocimiento", run_recognition, color="#1a3a6b").pack()

# ──────────────────────────────────────────────
#  PANTALLA: GESTIÓN DE USUARIOS
# ──────────────────────────────────────────────

def open_users_screen():
    win = Toplevel(root)
    win.title("Gestion de usuarios")
    win.geometry("540x400")
    win.configure(bg=BG)
    win.resizable(False, False)

    make_header(win, "Usuarios en BD")
    spacer(win)

    frame_list = Frame(win, bg=BG)
    frame_list.pack(fill=BOTH, expand=True, padx=20)

    scrollbar = Scrollbar(frame_list)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(frame_list, bg="#202020", fg=WHITE, selectbackground="#27ae60",
                      font=(FONT, 11), width=50, yscrollcommand=scrollbar.set,
                      relief=FLAT, borderwidth=0)
    listbox.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    status = status_lbl(win)

    def refresh():
        listbox.delete(0, END)
        users = db.getAllUsers()
        if not users:
            listbox.insert(END, "  (sin usuarios en la base de datos)")
        for u in users:
            listbox.insert(END, f"  #{u['id']:03d}  --  {u['name']}")
        status.config(text=f"{len(users)} usuario(s) en la BD.", fg=YELLOW)

    def delete_selected():
        sel = listbox.curselection()
        if not sel:
            return
        line = listbox.get(sel[0])
        name = line.split("--")[-1].strip()
        if msg.askyesno("Eliminar", f"Eliminar '{name}' de la BD?\n"
                                    "Las imagenes locales NO se borraran."):
            res = db.deleteUser(name)
            status.config(
                text=f"'{name}' eliminado." if res["affected"] else "No se pudo eliminar.",
                fg=GREEN if res["affected"] else RED)
            refresh()

    spacer(win)
    btn_frame = Frame(win, bg=BG)
    btn_frame.pack()
    styled_btn(btn_frame, "Actualizar", refresh,          color="#1a3a6b", w=14).pack(side=LEFT, padx=6)
    styled_btn(btn_frame, "Eliminar",   delete_selected,  color="#6b1a1a", w=14).pack(side=LEFT, padx=6)
    spacer(win)
    refresh()

# ──────────────────────────────────────────────
#  PANTALLA PRINCIPAL
# ──────────────────────────────────────────────

root = Tk()
root.title("Sistema de Asistencia Facial")
root.geometry("500x480")
root.configure(bg=BG)
root.resizable(False, False)

make_header(root, "Sistema de Asistencia Facial")
spacer(root)

styled_btn(root, "Registrar / Capturar rostros", open_capture_screen).pack()
spacer(root)
styled_btn(root, "Entrenar modelo",              open_train_screen).pack()
spacer(root)
styled_btn(root, "Reconocimiento en vivo",       open_recognition_screen).pack()
spacer(root)
styled_btn(root, "Gestion de usuarios (BD)",     open_users_screen).pack()
spacer(root)

info_frame = Frame(root, bg=BG)
info_frame.pack(fill=X, padx=20)

db_status_lbl = Label(info_frame, text="Verificando BD...", fg=YELLOW, bg=BG, font=(FONT, 10))
db_status_lbl.pack()

model_lbl = Label(info_frame, text="", fg=YELLOW, bg=BG, font=(FONT, 10))
model_lbl.pack()

def refresh_status():
    check_db_status(db_status_lbl)
    m = load_model()
    if m:
        model_lbl.config(text=f"Modelo listo  --  {len(m)} persona(s)", fg=GREEN)
    else:
        model_lbl.config(text="Sin modelo entrenado", fg=YELLOW)
    root.after(4000, refresh_status)

root.after(500, refresh_status)
root.mainloop()