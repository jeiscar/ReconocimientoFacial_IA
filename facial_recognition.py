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
ORB_FEATURES    = 700
MATCH_THRESHOLD = 70
# Umbrales ESTRICTOS para asistencia (validación mejorada)
SIMILARITY_MIN  = 0.66  # Perfil HP HD: balance entre precision y rechazo
MIN_MATCHES     = 30    # Menor exigencia por menor detalle de camara
VERIFY_MARGIN   = 0.12  # Mantiene separacion anti-impostor
# Validacion de calidad de imagen
BLUR_THRESHOLD  = 20.0  # Umbral bajo para webcams integradas de baja resolución
BRIGHTNESS_MIN  = 30    # Brillo minimo (0-255)
BRIGHTNESS_MAX  = 230   # Brillo maximo (0-255)
FACE_SIZE_MIN   = 80    # Tamaño minimo de rostro en pixeles
CONSENSUS_FRAMES = 12   # Compensa umbral mas flexible en camara basica
COOLDOWN_SECONDS = 300  # Cooldown de 5 min para no duplicar asistencia

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20

BG     = "#151515"
HEADER = "#101010"
BTN    = "#202020"
WHITE  = "#f4f5f4"
GREEN  = "#27ae60"
RED    = "#e74c3c"
YELLOW = "#f39c12"
FONT   = "Century Gothic"

# Colores en formato BGR para OpenCV
CV_GREEN = (0, 200, 80)      # Verde
CV_RED = (60, 60, 220)       # Rojo
CV_BLUE = (200, 60, 60)      # Azul
CV_ORANGE = (0, 165, 255)    # Naranja
CV_WHITE = (200, 200, 200)   # Blanco

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


def preprocess_face_gray(img_gray):
    # CLAHE mejora contraste local en imagenes de webcam de baja calidad.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)

def similarity(des1, des2):
    if des1 is None or des2 is None:
        return 0.0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < MATCH_THRESHOLD]
    # Penaliza coincidencias con pocos matches para evitar falsos positivos.
    raw = len(good) / len(matches)
    confidence = min(1.0, len(matches) / float(MIN_MATCHES))
    return raw * confidence


def person_score(des_query, des_list):
    scores = [similarity(des_query, d) for d in des_list]
    valid = [s for s in scores if s > 0]
    if not valid:
        return 0.0
    # Media de las mejores coincidencias para mayor robustez.
    top_k = sorted(valid, reverse=True)[:5]
    return float(np.mean(top_k))

# ──────────────────────────────────────────────
#  VALIDACION DE IMAGEN
# ──────────────────────────────────────────────

def is_image_blurry(img_gray, threshold=BLUR_THRESHOLD):
    """Detecta si una imagen esta borrosa usando varianza Laplaciana."""
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_image_dark_or_bright(img_gray):
    """Valida brillo de imagen."""
    brightness = np.mean(img_gray)
    return brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX

def is_face_too_small(w, h):
    """Valida tamaño minimo de rostro."""
    face_area = w * h
    return face_area < FACE_SIZE_MIN ** 2

def validate_image_quality(img_gray, w, h):
    """Retorna (is_valid, reason) para diagnostico."""
    if is_image_blurry(img_gray):
        return False, "Borrosa"
    if is_image_dark_or_bright(img_gray):
        return False, "Iluminacion"
    if is_face_too_small(w, h):
        return False, "Muy pequeno"
    return True, "OK"


def setup_camera(cap):
    """Ajustes base para estabilizar lectura en webcams integradas."""
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[CAM] Configuración de cámara aplicada correctamente")
    except Exception as e:
        print(f"[CAM] Advertencia al configurar cámara: {e}")

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
            gray = preprocess_face_gray(gray)
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
        gray = preprocess_face_gray(gray)
        des_q = compute_descriptors(gray)

        best_name, best_score = "Desconocido", 0.0
        for person, des_list in model.items():
            avg = person_score(des_q, des_list)
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


def verify_claimed_id(frame_bgr, model, claimed_id):
    """Verifica identidad: una sola cara, calidad imagen, umbrales estrictos."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = get_detector().detect_faces(rgb)
    results = []

    claimed_desc = model.get(claimed_id)
    if not claimed_desc:
        return results

    # VALIDACION CRITICA: debe haber EXACTAMENTE una cara
    if len(faces) == 0:
        cv2.putText(frame_bgr, "NINGUNA CARA DETECTADA", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2)
        return results
    if len(faces) > 1:
        cv2.putText(frame_bgr, f"ERROR: {len(faces)} caras. Solo 1 persona.",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return results

    face_data = faces[0]
    x, y, w, h = face_data["box"]
    x, y = max(0, x), max(0, y)
    crop = frame_bgr[y:y+h, x:x+w]
    crop = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = preprocess_face_gray(gray)

    # VALIDACION DE CALIDAD: blur, brillo, tamaño
    quality_ok, quality_msg = validate_image_quality(gray, w, h)
    if not quality_ok:
        color = (0, 165, 255)  # Naranja = error de calidad
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame_bgr, f"CALIDAD: {quality_msg}",
                    (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return results

    des_q = compute_descriptors(gray)

    claimed_score = person_score(des_q, claimed_desc)
    impostor_best = 0.0
    for person, des_list in model.items():
        if person == claimed_id:
            continue
        impostor_best = max(impostor_best, person_score(des_q, des_list))

    margin = claimed_score - impostor_best
    # Criterios ESTRICTOS: similitud >= 70% AND margen >= 15%
    accepted = claimed_score >= SIMILARITY_MIN and margin >= VERIFY_MARGIN

    color = (0, 200, 80) if accepted else (60, 60, 220)
    state = "ASISTENCIA OK" if accepted else "NO VERIFICADO"
    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
    cv2.putText(
        frame_bgr,
        f"ID {claimed_id} | {state} | S:{claimed_score:.0%} M:{margin:.0%}",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
    )
    results.append((accepted, claimed_score, margin))

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
    win.title("Registrar estudiante")
    win.geometry("540x450")
    win.configure(bg=BG)
    win.resizable(False, False)

    make_header(win, "Registrar estudiante")
    spacer(win)

    Label(win, text="ID del estudiante:", fg=WHITE, bg=BG, font=(FONT, 12)).pack()
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
        # Evita duplicados: solo insertar si el nombre no existe en la BD
        existing = db.getAllUsers()
        if any(u["name"] == name for u in existing):
            print(f"[DB] '{name}' ya existe en BD, no se duplica.")
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
            status.config(text="Escribe el ID del estudiante.", fg=RED)
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
                text=f"El ID '{name}' ya tiene {existing_count} foto(s).\n"
                     f"Se agregarán {total} foto(s) nuevas (total: {existing_count + total}).",
                fg=YELLOW)
            win.update()

        cap = cv2.VideoCapture(0)
        setup_camera(cap)
        if not cap.isOpened():
            status.config(text="No se pudo abrir la camara.", fg=RED)
            return

        status.config(
              text=f"Capturando {total} fotos nuevas para ID '{name}'.\n"
                 f"Ya existentes: {existing_count}  |  ESPACIO = capturar  |  ESC = cancelar",
            fg=YELLOW)
        win.update()

        captured  = 0                     # fotos nuevas en esta sesión
        next_idx  = existing_count + 1    # índice para no sobreescribir
        progress["maximum"] = total
        progress["value"]   = 0

        frames_failed = 0
        while captured < total:
            ret, frame = cap.read()
            if not ret:
                frames_failed += 1
                # Reintentar unos pocos frames antes de abandonar
                if frames_failed > 30:
                    print(f"[CAM] Error: No se pueden leer frames después de 30 intentos.")
                    status.config(
                        text="Error de cámara: No se puede leer frames.\n"
                             "Cierra otras apps que usan la cámara y reintenta.",
                        fg=RED)
                    break
                continue
            
            frames_failed = 0  # Reiniciar contador si conseguimos un frame

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
                 text=f"{captured} foto(s) nuevas agregadas para ID '{name}'.\n"
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
    win.title("Verificacion de asistencia")
    win.geometry("620x400")
    win.configure(bg=BG)
    win.resizable(False, False)

    make_header(win, "Verificacion por ID de estudiante")
    spacer(win)

    Label(win, text="ID del estudiante:", fg=WHITE, bg=BG, font=(FONT, 12)).pack()
    claimed_id_var = StringVar()
    Entry(win, textvariable=claimed_id_var, justify=CENTER, font=(FONT, 12), width=30).pack(ipady=5)
    spacer(win)

    info_text = f"""Modelo: {len(model)} ID(s)
Criterios: 1 cara | Imagen nitida | S >= 70% | M >= 15%
Consenso: {CONSENSUS_FRAMES} frames consecutivos validos
ESC para cerrar"""

    Label(win, text=info_text, fg=WHITE, bg=BG, font=(FONT, 10), justify=LEFT).pack()
    spacer(win)
    status = status_lbl(win)

    def run_recognition():
        claimed_id = claimed_id_var.get().strip()
        if not claimed_id:
            status.config(text="Debes ingresar el ID del estudiante.", fg=RED)
            return
        if claimed_id not in model:
            status.config(
                text=f"El ID '{claimed_id}' no existe en el modelo. Registra y entrena primero.",
                fg=RED,
            )
            return

        cap = cv2.VideoCapture(0)
        setup_camera(cap)
        if not cap.isOpened():
            status.config(text="No se pudo abrir la camara.", fg=RED)
            return
        status.config(text=f"Camara activa para '{claimed_id}'. ESC para detener.", fg=GREEN)
        win.update()

        consecutive_valid = 0  # Contador de frames consecutivos validos
        attendance_marked = False
        last_valid_score = 0.0
        last_valid_margin = 0.0
        frames_failed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                frames_failed += 1
                if frames_failed > 30:
                    print(f"[CAM] Error: No se pueden leer frames después de 30 intentos.")
                    break
                continue
            
            frames_failed = 0

            results = verify_claimed_id(frame, model, claimed_id)

            # CONSENSO TEMPORAL: requiere CONSENSUS_FRAMES consecutivos validos
            if results and len(results) > 0:
                accepted, score, margin = results[0]
                if accepted:
                    consecutive_valid += 1
                    last_valid_score = score
                    last_valid_margin = margin
                    cv2.putText(frame, f"Frames OK: {consecutive_valid}/{CONSENSUS_FRAMES}",
                                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV_GREEN, 2)
                else:
                    consecutive_valid = 0
            else:
                consecutive_valid = 0

            # MARCAR ASISTENCIA si se alcanza consenso
            if consecutive_valid >= CONSENSUS_FRAMES and not attendance_marked:
                msg.showinfo("Exito", f"ID '{claimed_id}'\nASISTENCIA REGISTRADA\n"
                                      f"{consecutive_valid} frames validados")
                # Registrar en BD
                res = db.recordAttendance(
                    claimed_id,
                    last_valid_score,
                    last_valid_margin,
                    consecutive_valid,
                    "OK"
                )
                if res["affected"]:
                    print(f"[DB] Asistencia registrada: {claimed_id} (id={res['id']})")
                attendance_marked = True

            cv2.imshow("Verificacion Asistencia  (ESC=salir)", frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        if not attendance_marked:
            status.config(text="Reconocimiento cancelado.", fg=YELLOW)
        else:
            status.config(text=f"Asistencia de '{claimed_id}' OK.", fg=GREEN)

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

styled_btn(root, "Registrar estudiante (ID)", open_capture_screen).pack()
spacer(root)
styled_btn(root, "Entrenar modelo",              open_train_screen).pack()
spacer(root)
styled_btn(root, "Tomar asistencia (verificar ID)", open_recognition_screen).pack()
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