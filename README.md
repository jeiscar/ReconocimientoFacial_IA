# Sistema de Reconocimiento Facial para Control de Asistencia

Sistema avanzado de registro y autenticación de usuarios mediante reconocimiento facial, diseñado para la toma de asistencia de estudiantes. Utiliza inteligencia artificial (MTCNN + ORB) para detectar y comparar rostros con alta precisión.

## 📋 Descripción General

Aplicativo de escritorio que permite:
- **Registro de usuarios** con múltiples capturas faciales (5-200 fotos por usuario)
- **Entrenamiento automático** del modelo con descriptores ORB
- **Autenticación biométrica** por reconocimiento facial en tiempo real
- **Almacenamiento dual**: dataset local + referencia en MySQL
- **Interfaz gráfica intuitiva** con Tkinter
- **Reconocimiento en vivo** con visualización de resultados

## ✨ Características Principales

- **Detección de rostros robusta** en tiempo real con MTCNN (Multi-task Cascaded CNN)
- **Extracción de características** con ORB (Oriented FAST and Rotated BRIEF)
- **Entrenamiento incremental** del modelo sin necesidad de GPU
- **Múltiples imágenes por usuario** (por defecto 30, configurable 5-200)
- **Similitud adaptativa** por usuario (0.66 de umbral configurado)
- **Almacenamiento eficiente**: archivos locales + BD con referencia
- **Manejo robusto** de conexiones MySQL con cierre seguro
- **Feedback visual** en tiempo real durante captura y reconocimiento

## 🛠️ Stack Tecnológico

| Componente | Versión | Propósito |
|-----------|---------|-----------|
| **Python** | 3.11+ | Lenguaje de programación |
| **Tkinter** | Built-in | Interfaz gráfica multiplataforma |
| **OpenCV (cv2)** | 4.8+ | Procesamiento de imágenes, visualización |
| **MTCNN** | 0.1.1 | Detección precisa de rostros (RNN-CNN) |
| **NumPy** | 1.24+ | Operaciones numéricas y arrays |
| **Matplotlib** | 3.7+ | Lectura/manipulación de imágenes |
| **TensorFlow** | 2.13+ | Backend para MTCNN |
| **MySQL Connector** | 8.0+ | Conexión y operaciones en MySQL |
| **Pickle** | Built-in | Serialización del modelo entrenado |

### Justificación Técnica

- **MTCNN:** Red neuronal de 3 etapas (P-Net, R-Net, O-Net) para detección facial con excelente precision/recall. Mejor que Haar Cascades para rostros en ángulos y oclusiones parciales.
- **ORB:** Algoritmo rápido y sin licencia para extraer descriptores (keypoints + BRIEF). No requiere GPU, ideal para sistemas sin recursos.
- **Pickle:** Serialización segura del diccionario de descriptores entrenados (tamaño ~1-50 MB según cantidad de usuarios/imágenes).
- **MySQL:** Base de datos relacional para persistencia y auditoría, almacena referencia de foto por usuario en LONGBLOB.

## 📦 Requisitos Previos

- **Python 3.11+** (probado en 3.11, 3.12)
- **MySQL 5.7+** o MariaDB 10.4+
- **Webcam funcional** 
- **Permisos de escritura** en carpetas `dataset/`, `model/`, `temp/`
- **SO:** Windows, Linux o macOS (cualquiera con Python 3.11+)

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/jeiscar/ReconocimientoFacial_IA.git
cd ReconocimientoFacial_IA
```

### 2. Crear y activar entorno virtual

**Windows (PowerShell/CMD):**
```bash
py -3.11 -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install opencv-python numpy matplotlib mtcnn tensorflow mysql-connector-python
```

**Nota:** TensorFlow puede tardar varios minutos la primera instalación. Se descargará ~500MB.

### 4. Configurar la base de datos

#### a) Crear base de datos en MySQL:
```bash
mysql -u root -p < script.sql
```

O ejecutar manualmente:
```sql
CREATE DATABASE IF NOT EXISTS agency CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE agency;

CREATE TABLE IF NOT EXISTS user(
    idUser INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    photo LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHARSET=utf8mb4
);
```

#### b) Guardar credenciales en `keys.json`:
```json
{
    "host": "localhost",
    "user": "root",
    "password": "tu_contraseña",
    "database": "agency",
    "autocommit": true
}
```

⚠️ **IMPORTANTE:** `keys.json` está en `.gitignore` por seguridad. **Nunca versionar credenciales.**

#### c) Verificar conexión:
```bash
python -c "import database as db; ok, msg = db.testConnection(); print(msg)"
```

### 5. Estructura de directorios (auto-creada)
```
ReconocimientoFacial_IA/
├── dataset/              # Carpetas con imágenes de entrenamiento
│   ├── usuario1/        # Subcarpeta por usuario
│   │   ├── usuario1_0001.jpg
│   │   ├── usuario1_0002.jpg
│   │   └── ...
│   └── usuario2/
│       ├── usuario2_0001.jpg
│       └── ...
├── model/               # Modelos entrenados
│   └── orb_model.pkl    # Descriptores ORB serializados
├── temp/                # Archivos temporales (limpiados al salir)
├── img/                 # (Deprecated) Para compatibilidad
├── facial_recognition.py # Aplicación principal (GUI + lógica)
├── database.py          # Funciones de conexión y operaciones BD
├── keys.json            # Credenciales (NO versionado)
├── script.sql           # Script de creación de BD
└── README.md            # Documentación
```

## ✅ Criterios de Validación Mejorados (v2.1)

Para garantizar **máxima precisión y seguridad** en la toma de asistencia, el sistema implementa múltiples capas de validación:

### 1️⃣ **Validación Estructural**

| Criterio | Valor | Propósito |
|----------|-------|----------|
| **Caras permitidas** | Exactamente 1 | Evita suplantación con otros en frame |
| **Error si múltiples** | > 1 rostro | Bloquea reconocimiento en grupo |
| **Error si ninguna** | 0 rostros | Previene validación sin cara |

### 2️⃣ **Validación de Calidad de Imagen**

```
Test de blur (Laplaciano variance):
  ✓ OK: > 20.0  (imagen aceptable para webcam integrada)
  ✗ FALLA: < 20  (imagen borrosa)
  
Test de brillo:
  ✓ OK: 30-230 (rango aceptable)
  ✗ FALLA: < 30  (muy oscura)
  ✗ FALLA: > 230 (muy clara/saturada)
  
Test de tamaño de rostro:
  ✓ OK: > 80×80 px (rostro grande)
  ✗ FALLA: < 80×80 (rostro muy pequeño)
```

**Indicador visual:**
- 🟢 Verde: Criterios OK, esperando consenso
- 🟠 Naranja: Error de calidad (especifica motivo)
- 🔴 Rojo: Múltiples caras / No identificado
- 🔵 Azul: No hay cara

### 3️⃣ **Umbrales de Similitud (Perfil HP HD Camara)**

```
Similitud (S):       >= 66%   [perfil camara basica]
Margen anti-impostor (M): >= 12%
Min. coincidencias:  >= 30
```

**Fórmula de aceptación:**
```
ASISTENCIA_OK = (Similitud >= 66%) AND (Margen >= 12%) AND (Caras == 1) AND (Calidad = OK)
```

### 4️⃣ **Consenso Temporal (Anti-Ruido)**

```
Requeridos: 12 frames consecutivos válidos
Tiempo aprox: 0.3-0.5 segundos a 30 FPS
Ventaja: 1 frame falso no marca asistencia
```

### 4.1️⃣ Recomendaciones para webcam HP HD

- Usar luz frontal fija (evitar contraluz de ventana).
- Distancia recomendada de rostro: 40-70 cm.
- Mantener el rostro centrado y estable por ~1 segundo.
- Capturar registro con 50-80 fotos por estudiante en distintos dias.
- Reentrenar el modelo despues de agregar nuevas fotos.

**Flujo:**
1. Frame 1 válido → contador = 1
2. Frame 2 válido → contador = 2
3. ... (continúa si todos válidos)
4. Frame 12 válido → **ASISTENCIA REGISTRADA**
5. Si un frame falla → contador se reinicia a 0

### 5️⃣ **Auditoría Completa en Base de Datos**

Cada asistencia registrada contiene:
```sql
INSERT INTO attendance (
  student_id,        -- ID ingresado por usuario
  similarity_score,   -- 70% ej.
  margin_score,       -- 15% ej.
  frames_consensus,   -- 10 frames
  quality_check,      -- "OK"
  status,             -- "VERIFIED"
  timestamp           -- 2024-03-24 15:32:05
)
```

**Beneficios:**
- Trazabilidad total
- Revisión de resultados cuestionables
- Análisis histórico de precisión
- Detectar patrones de fallos

---

## 📊 Protocolo de Calibración Recomendado

Antes de usar en producción, valida precisión con tus estudiantes:

### Paso 1: Dataset de Validación
```
Por cada estudiante capturar:
- 50-80 fotos en registro (variación de luz, gafas, expresión)
- 10-15 fotos de prueba (no usadas en entrenamiento)
- Entrena con las primeras 50
```

### Paso 2: Medir Métricas
```
FAR (False Acceptance Rate) = Impostores aceptados / Total impostores
FRR (False Rejection Rate) = Legítimos rechazados / Total legítimos
EER (Equal Error Rate) = punto donde FAR = FRR (objetivo < 2%)
```

### Paso 3: Ajustar si es necesario
```
Si FAR alto (muchos falsos positivos):
  → Aumentar SIMILARITY_MIN a 0.75 o 0.80
  → Aumentar VERIFY_MARGIN a 0.18 o 0.20

Si FRR alto (muchos falsos negativos):
  → Reducir SIMILARITY_MIN a 0.65
  → Aumentar cantidad de fotos por estudiante a 80-100
  → Mejorar iluminación de captura
```

### Ejecutar la aplicación
```bash
python facial_recognition.py
```

La interfaz mostrará:
- **Estado de BD:** conexión a MySQL
- **Botones principales:** Registrar, Reconocer, Entrenar, Gestionar
- Todas las peticiones de la GUI se procesan en la BD de forma sincróna

### 🔹 Flujo 1: Registrar Nuevo Usuario

1. **Clic en "Registrar persona"** → Se abre ventana de captura
2. **Ingresa nombre del estudiante** (ej: "Juan García")
3. **Configura cantidad de fotos** (por defecto 30, rango 5-200)
4. **Presiona "Iniciar captura"** → Se abre la webcam
5. **Presiona ESPACIO** para capturar cada foto (se ve en pantalla: "Nuevas: X/30 | Total acumulado: Y")
6. **Presiona ESC** para finalizar
7. **Sistema guarda automáticamente:**
   - ✓ Imágenes en carpeta `dataset/nombre_usuario/`
   - ✓ Primera imagen como referencia en MySQL (LONGBLOB)
   - ✓ Se prepara para entrenar

**Notas importantes:**
- Si el usuario ya tiene 20 fotos y vuelves a registrar con 30, al final tendrá 50 fotos
- El sistema detecta rostros automáticamente; sin rostro detectado muestra advertencia
- Las fotos se nombran: `nombre_usuario_0001.jpg`, `nombre_usuario_0002.jpg`, etc.
- Si presionas ESPACIO sin rostro detectado, no se captura nada

### 🔹 Flujo 2: Entrenar el Modelo

1. **Clic en "Entrenar Modelo"** → Sistema analiza todas las imágenes
2. **Proceso:**
   - Lee todas las carpetas en `dataset/`
   - Por cada usuario y cada imagen:
     - Convierte a escala de grises
     - Extrae descriptores ORB (700 características por imagen)
     - Guarda descriptores en diccionario por usuario
   - Serializa todo en `model/orb_model.pkl` (pickle)
3. **Resultado:** "Modelo entrenado: 5 persona(s), 145 imagen(es)"
4. **El modelo estará listo para predicción**

**Detalles técnicos:**
- Cada imagen produce hasta 700 descriptores (ORB)
- Cada usuario tendrá lista de descriptores: `[desc_img1, desc_img2, ..., desc_imgN]`
- Total datos del modelo = usuarios × imágenes × 700 características
- Tamaño típico: 2-5 usuarios, 30 imágenes c/u = ~5 MB archivo .pkl

### 🔹 Flujo 3: Tomar Asistencia (Verificación por ID)

1. **Clic en "Tomar asistencia (verificar ID)"** → Se abre ventana de verificación
2. **Ingresa el ID del estudiante** en el campo de texto
3. **Presiona "Iniciar reconocimiento"** → Se activa la webcam
4. **Sistema aplica verificación 1:1 estricta:**
   - Solo acepta exactamente **1 cara** en el frame
   - Valida calidad: blur (`< 20`), brillo (`30-230`), tamaño rosotro (`> 80px`)
   - Extrae descriptores ORB y compara contra el ID reclamado
   - Calcula similitud (`S`) y margen anti-impostor (`M = S_claimed - S_best_other`)
   - Acepta solo si `S >= 66%` AND `M >= 12%`
5. **Consenso temporal:** requiere 12 frames consecutivos válidos
6. **Al alcanzar consenso:**
   - ✓ Verde: `ID XXXX | ASISTENCIA OK | S:XX% M:XX%`
   - Se muestra popup de confirmación
   - Se registra en tabla `attendance` de la BD (score, margen, frames, timestamp)
7. **Presiona ESC** para salir

> ⚠️ El sistema **no tiene liveness detection**. Una foto impresa podría pasar la validación si tiene calidad suficiente. Para entornos críticos se recomienda agregar detección de parpadeo (blink detection).

### 🔹 Flujo 4: Gestionar Usuarios

**Listar todos los usuarios:**
1. Clic en "Listar Usuarios" → Muestra tabla con ID, Nombre
2. Los datos vienen directamente de MySQL

**Eliminar un usuario:**
1. Clic en "Eliminar Usuario"
2. Ingresa el nombre exacto
3. Se elimina de BD (archivo en `dataset/` se puede limpiar manualmente)

### 📊 Configuración Avanzada

Edita `facial_recognition.py` para personalizar parámetros:

```python
# Tamaño de rostro recortado (píxeles)
IMG_SIZE = (150, 200)

# Cantidad por defecto de fotos a capturar
DEFAULT_PHOTOS = 30

# Características de ORB a extraer
ORB_FEATURES = 700

# Umbral de distancia en matching ORB
MATCH_THRESHOLD = 70

# Similitud mínima para considerar como reconocido
SIMILARITY_MIN = 0.66  # 66%
```

**Ajustes recomendados:**
- **Aumentar SIMILARITY_MIN a 0.75** si hay falsos positivos
- **Reducir a 0.55** si hay muchos falsos negativos (y más fotos por usuario)
- **Aumentar ORB_FEATURES a 1000** para mayor precisión (pero más lento)
- **IMG_SIZE más grande** = mejor información pero más lentitud

## 📁 Estructura del Proyecto

```
ReconocimientoFacial_IA/
│
├── facial_recognition.py  # Aplicación principal (GUI + lógica ML)
├── database.py            # Gestor de conexión MySQL
├── restore_db.py          # Utilidad: re-registra usuarios desde dataset/ si la BD fue limpiada
├── script.sql             # Script de creación de BD
├── keys.json              # Credenciales (gitignored)
├── README.md              # Documentación
│
├── dataset/               # Imágenes de entrenamiento organizadas por usuario
│   ├── usuario1/
│   │   ├── usuario1_0001.jpg
│   │   ├── usuario1_0002.jpg
│   │   └── ...
│   └── usuario2/
│       └── usuario2_0001.jpg
│
├── model/                 # Modelos serializados
│   └── orb_model.pkl      # Descriptores ORB entrenados (pickle)
│
├── temp/                  # Archivos temporales
│
├── img/                   # Carpeta heredada (deprecated)
│
├── venv/                  # Entorno virtual (gitignored)
│
└── __pycache__/           # Cache Python (gitignored)
```

## 🔬 Arquitectura Técnica del Sistema

### 1️⃣ **Fase 1: Registro de Imágenes (Captura)**

```
┌─────────────────────┐
│  Input: Nombre      │
│  + Cantidad fotos   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Abrir webcam (cv2.VideoCapture)     │
│ Loop: ESPACIO=captura, ESC=salir    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ MTCNN.detect_faces(frame RGB)       │
│ → Encuentra bounding box del rostro │
└──────────┬──────────────────────────┘
           │
           ├─ No hay rostro: Mostrar advertencia
           │
           ▼ (ESPACIO presionado)
┌─────────────────────────────────────┐
│ detect_and_crop(frame)              │
│ - Recortar región de interés        │
│ - cv2.resize(crop, (150,200))       │
│ - cv2.imwrite(dataset/user/...)     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Guardar en disco local              │
│ (Primera imagen → DB como referencia│
└─────────────────────────────────────┘
```

**Código clave:**
```python
def detect_and_crop(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = get_detector().detect_faces(rgb)
    if not faces:
        return None
    
    x, y, w, h = faces[0]["box"]  # Mayor rostro en frame
    crop = frame_bgr[y:y+h, x:x+w]
    return cv2.resize(crop, (150, 200), interpolation=cv2.INTER_CUBIC)
```

**Almacenamiento dual:**
- 📁 Local: `dataset/nombre_usuario/nombre_usuario_NNNN.jpg`
- 💾 BD: Primera imagen guardada como LONGBLOB en tabla `user`

---

### 2️⃣ **Fase 2: Entrenamiento del Modelo**

```
┌──────────────────────────────────────┐
│ Cargar todas las carpetas en         │
│ dataset/ (cada carpeta = usuario)    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ Para cada usuario:                           │
│  - Listar todas las imágenes (.jpg, .png)    │
│  - Por cada imagen:                          │
│    a) Leer en escala de grises               │
│    b) orb.detectAndCompute(gray)             │
│       → Extraer 500 descriptores             │
│    c) Guardar descriptores en lista          │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ Modelo = {                                    │
│   "Juan": [desc_img1, desc_img2, ...],      │
│   "María": [desc_img1, desc_img2, ...],     │
│   ...                                         │
│ }                                             │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ pickle.dump(modelo, model/orb_model.pkl)     │
│ Archivo serializado (~5-50 MB)               │
└──────────────────────────────────────────────┘
```

**Parámetros de extracción:**
```python
orb = cv2.ORB_create(nfeatures=700)  # 700 características por imagen
kp, descriptors = orb.detectAndCompute(gray_image, None)
# kp: hasta 700 keypoints (x, y, ángulo, escala)
# descriptors: shape (N, 32) - cada descriptor = 256 bits
```

**Tamaño del modelo ejemplos:**
- 2 usuarios × 30 imágenes c/u = 3,000 descriptores × 32 bytes = ~96 KB
- 10 usuarios × 50 imágenes c/u = 500,000 descriptores × 32 bytes = ~16 MB

---

### 3️⃣ **Fase 3: Reconocimiento en Vivo (Predicción)**

```
┌────────────────────────────────────────┐
│ pickle.load(model/orb_model.pkl)       │
│ Cargar descriptor de cada usuario      │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ cv2.VideoCapture(0) + Loop en tiempo real          │
│ Cada frame:                                         │
│  1. MTCNN.detect_faces() → encontrar rostros      │
│  2. Para cada rostro detectado:                     │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ detect_and_crop() → rostro normalizado (150×200)  │
│ gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)     │
│ des_query = orb.detectAndCompute(gray)            │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ Para cada usuario en modelo:                       │
│  - BFMatcher + NORM_HAMMING                        │
│  - Para cada imagen almacenada del usuario:        │
│    • matches = comparar(des_query, des_stored)    │
│    • good = [m for m in matches if dist < 70]     │
│    • similitud_img = len(good) / len(matches)     │
│  - promedio_usuario = mean(similitudes)           │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ best_user = argmax(promedios)                      │
│ if best_score < SIMILARITY_MIN (0.66):            │
│     → "Desconocido"                               │
│ else:                                              │
│     → "Nombre_Usuario (XX%)"                      │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ Dibujar en pantalla:                               │
│ - Rectángulo (verde o rojo)                        │
│ - Nombre + porcentaje                              │
│ - Repetir cada frame (~30 FPS)                     │
└────────────────────────────────────────────────────┘
```

**Función de similitud:**
```python
def similarity(des1, des2):
    """Compara dos conjuntos de descriptores ORB"""
    if des1 is None or des2 is None:
        return 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if not matches:
        return 0.0
    
    # Filtrar coincidencias "buenas" (distancia < 70)
    good = [m for m in matches if m.distance < 70]
    
    # Ratio de similitud = buenas / totales
    return len(good) / len(matches)

# Ejemplo:
# 150 matches totales, 100 con distancia < 70
# → similitud = 100/150 = 0.667 = 66.7%
```

---

### 4️⃣ **Base de Datos MySQL**

**Esquema:**
```sql
CREATE TABLE user (
    idUser INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    photo LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance (
    idAttendance INT AUTO_INCREMENT PRIMARY KEY,
    student_id   VARCHAR(100) NOT NULL,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    similarity_score  FLOAT NOT NULL,
    margin_score      FLOAT NOT NULL,
    frames_consensus  INT NOT NULL,
    quality_check     VARCHAR(50),
    status ENUM('VERIFIED', 'REJECTED', 'QUALITY_ERROR') DEFAULT 'VERIFIED',
    INDEX idx_student_id (student_id),
    INDEX idx_timestamp  (timestamp)
);
```

**Operaciones:**

| Operación | Función | Datos |
|-----------|---------|--------|
| Guardar referencia | `registerUser(name, photo_path)` | Inserta 1ª imagen como BLOB (sin duplicar) |
| Recuperar referencia | `getUser(name, output_path)` | Extrae foto de BD, escribe a disco |
| Listar usuarios | `getAllUsers()` | SELECT id, name (sin fotos) |
| Eliminar usuario | `deleteUser(name)` | DELETE (archivos en disco se limpian manualmente) |
| Verificar conexión | `testConnection()` | Conexión de prueba sin escritura |
| Registrar asistencia | `recordAttendance(student_id, score, margin, frames)` | INSERT en tabla `attendance` |
| Asistencia hoy | `getAttendanceToday(student_id)` | SELECT por fecha actual |

**Manejo de conexiones (CRÍTICO):**
```python
def registerUser(name, photo):
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("INSERT INTO...")
        connection.commit()
    except db.Error as error:
        print(error)
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()  # ← CIERRE OBLIGATORIO
    return {"id": user_id, "affected": inserted}
```

⚠️ **Importante:** Sin cierre explícito, las conexiones se acumulan y MySQL rechaza nuevas conexiones.

---

### 5️⃣ **Interfaz Gráfica (Tkinter)**

**Arquitectura:**
```
main window (root)
├── Frame principal
│   ├── Label: Estado BD
│   ├── Button: "Registrar persona"
│   │   └── Toplevel: open_capture_screen()
│   ├── Button: "Entrenar Modelo"
│   ├── Button: "Reconocer"
│   │   └── Toplevel: recognition loop en vivo
│   └── Button: "Gestionar"
```

**Componentes clave:**
- `Tkinter.Toplevel` para ventanas secundarias
- `ttk.Progressbar` para mostrar captura de fotos
- `cv2.imshow()` para visualización de captura/reconocimiento
- Label dinámicos para estados y errores

## 🧠 Análisis de Machine Learning

### Tipo de Solución
**Clasificación:** Verificación biométrica (1:1 matching)
- **Método:** Extracción de características + Comparación por similitud
- **Entrada:** Imagen de rostro
- **Salida:** Identidad (usuario conocido o "Desconocido")
- **Modelo:** No supervisado independiente por usuario

### Algoritmo Implementado: ORB + BFMatcher

| Aspecto | Detalles |
|--------|---------|
| **Ventajas** | ✓ Rápido (5-10ms/imagen), ✓ Sin GPU necesario, ✓ Robusto a rotaciones menores, ✓ 256 bits por descriptor |
| **Desventajas** | - Sensible a iluminación, - Precisión ~85-92%, - Se degrada con ángulos extremos |
| **Precisión estimada** | 85-92% en condiciones normales |
| **Velocidad** | ~30 FPS en tiempo real |
| **Escalabilidad** | ✓ Lineal con usuarios × imágenes |

### Alternativas evaluadas:

| Algoritmo | Precisión | Velocidad | GPU | Complejidad | Recomendación |
|-----------|-----------|-----------|-----|-------------|---------------|
| **ORB (actual)** | 85-92% | Muy rápida | No | Baja | ✓ Producción ligera |
| **FaceNet + Embedding** | 99%+ | Media | Sí | Media | 🎯 Mejor precisión |
| **VGGFace2** | 98%+ | Lenta | Sí | Alta | Precisión máxima |
| **DeepFace** | 97%+ | Lenta | Sí | Alta | Librería simplificada |
| **Eigenfaces** | 80% | Muy rápida | No | Baja | Obsoleto |

### Dataset

**Características actuales:**
- **Tipo:** Dataset propio generado en tiempo real
- **Formato:** JPG, 150×200 píx, escala de grises
- **Almacenamiento:** Local en `dataset/usuario/` + referencia en MySQL
- **Cantidad típica:** 30-50 imágenes por usuario
- **Total típico:** 2-10 usuarios, 60-500 imágenes totales

**Riesgos identificados:**

| Riesgo | Impacto | Mitigación |
|--------|--------|-----------|
| **Variabilidad temporal** | Barba, gafas, peinado en otra sesión | Capturar múltiples imágenes, reentrenar |
| **Iluminación** | ORB sensible a cambios de luz | Normalización de histograma opcional |
| **Ángulo facial** | Rostro de perfil/inclinado | MTCNN tiene cierta robustez (~45°) |
| **Oclusión** | Barbijo, bufanda, sombrero | Requiere detección de múltiples puntos |
| **Datos personales** | Cumplimiento GDPR | Encriptación de foto en BD, consentimiento |
| **Falsos positivos** | Usuario A se parece a B | Aumentar umbral SIMILARITY_MIN o imágenes |

## 🎯 Mejoras y Upgrades Futuros

### Corto Plazo (v2.0)
- [ ] Interfaz web con Flask (no solo Tkinter)
- [ ] Logging de intentos fallidos y exitosos
- [ ] Exportar asistencia a CSV/Excel
- [ ] Reconocimiento multi-rostro simultáneo
- [ ] Base de datos: SQL query logs y auditoría

### Mediano Plazo (v3.0)
- [ ] Migrar a **FaceNet** para precisión 99%+
- [ ] Detección de vida (liveness detection) contra deepfakes
- [ ] Encriptación AES-256 de imágenes en BD
- [ ] API REST para integración con SIS
- [ ] Dashboard de estadísticas en tiempo real

### Largo Plazo (v4.0)
- [ ] Model serving con TensorFlow Serving
- [ ] Reconocimiento multi-modal (iris + rostro)
- [ ] Sincronización en cloud (Azure/AWS)
- [ ] Reconocimiento federado sin trasmitir datos biométricos

## 🐛 Solución de Problemas

### Error de Conexión MySQL

**Síntoma:** `Error: Can't connect to MySQL server at 'localhost'`

**Soluciones:**
```bash
# Verificar que MySQL está corriendo
# Windows:
Get-Service MySQL80  # o el nombre de tu servicio

# Linux/Mac:
brew services list | grep mysql

# Verificar credenciales en keys.json
cat keys.json

# Probar conexión manualmente
mysql -u root -p -h localhost

# Ver puerto (por defecto 3306)
netstat -an | grep 3306

# Firewall: Asegurar que puerto 3306 está abierto
```

### Error: "ModuleNotFoundError: No module named 'cv2'"

**Solución:**
```bash
# Activar entorno virtual y reinstalar
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstalar explicito
pip install --upgrade opencv-python
pip freeze | grep opencv
```

### Error: "No faces detected"

**Causas y soluciones:**

| Causa | Solución |
|-------|----------|
| Iluminación pobre | Aumentar luz frontal, evitar contraluz |
| Rostro muy pequeño en frame | Acercarse a la cámara |
| Ángulo extremo | Mirar hacia la cámara (máx 45° de ángulo) |
| Rostro parcialmente cubierto | Quitar buffanda, gafas de sol, barbijo |
| Problema de cámara | Probar con otra app (WhatsApp camera), reiniciar |

### La comparación siempre falla (Desconocido)

**Síntomas:**
- Usuario registrado pero reconocimiento dice "Desconocido (15%)"
- Similitud muy baja aunque sea la misma persona

**Pasos:**
1. **Verificar que el modelo está entrenado:**
   ```bash
   ls -la model/orb_model.pkl
   # Si no existe o está vacío, hacer click en "Entrenar Modelo"
   ```

2. **Aumentar cantidad de imágenes:**
   - Registrar 50-80 fotos en lugar de 30 (capturar en diferentes condiciones)
   - Reentrenar modelo

3. **Bajar umbral de similitud:**
   ```python
   # En facial_recognition.py
   SIMILARITY_MIN = 0.55  # (en lugar de 0.66 por defecto)
   ```

4. **Verificar que está en modo predicción:**
   - Entrar a "Reconocer" (no "Registrar")
   - Esperar a que el modelo cargue

### La BD está llena, datos viejos

**Limpiar usuarios:**
```bash
# Opción 1: Interfaz - Click "Gestionar" → "Eliminar Usuario"

# Opción 2: MySQL directo
mysql -u root -p agency
DELETE FROM user WHERE name = 'nombre_usuario';

# Opción 3: Borrar todo y recrear
DROP TABLE user;
# (Luego ejecutar script.sql nuevamente)
```

**Limpiar dataset local:**
```bash
# Borrar todas las carpetas de usuarios
rm -rf dataset/*

# Reentrenar (modelo quedará vacío)
```

### TensorFlow tarda mucho en la primera importación

**Comportamiento normal:** Primer import = 10-30 segundos (carga modelo MTCNN)

**Soluciones:**
```bash
# Verificar que TensorFlow esté instalado correctamente
pip show tensorflow

# Si falla:
pip uninstall tensorflow mtcnn -y
pip install tensorflow mtcnn
```

### Archivo keys.json no encontrado

**Solución:** Crear manualmente en raíz del proyecto:
```json
{
    "host": "localhost",
    "user": "root",
    "password": "tu_contraseña",
    "database": "agency",
    "autocommit": true
}
```

---

## 📊 Monitoreo y Logs

Para diagnosticar problemas, ejecutar el sistema con output detallado:

```python
# En terminal, antes de ejecutar
export PYTHONUNBUFFERED=1  # (Linux/Mac)

# Windows PowerShell
$env:PYTHONUNBUFFERED=1

# Luego ejecutar
python facial_recognition.py 2>&1 | tee debug.log

# Ver logs
tail -f debug.log
```

**Puntos de logging importantes:**
- `[DB]` → Operaciones de base de datos
- `[MTCNN]` → Detección de rostros
- `[Model]` → Entrenamiento y predicción
- `[OpenCV]` → Capturas y visualización

## � Referencias Técnicas

### Papers y Recursos
- MTCNN: [Joint Face Detection and Alignment using Multi-task CNN](https://arxiv.org/abs/1604.02878)
- ORB: [ORB: An Efficient Alternative to SIFT or SURF](https://arxiv.org/abs/1508.04721)
- OpenCV: [detectAndCompute Documentation](https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html)
- FaceNet: [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)

### Métricas de Evaluación

```python
# Precisión de reconocimiento
TP = True Positives (reconocido correctamente)
FP = False Positives (reconocido incorrectamente como otro)
TN = True Negatives (rechazado correctamente)
FN = False Negatives (rechazado siendo la persona correcta)

Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)  # De los identificados, ¿cuántos eran correctos?
Recall = TP / (TP + FN)     # De los verdaderos, ¿cuántos se identificaron?
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

---

## 📋 Roadmap

### ✅ v2.1 (Actual)
- ✓ Registro multi-imagen (5-200 fotos, sin duplicados en BD)
- ✓ Entrenamiento ORB incremental (700 features, CLAHE + GaussianBlur)
- ✓ Verificación 1:1 por ID de estudiante (no reconocimiento abierto)
- ✓ Validación de calidad: blur / brillo / tamaño de rostro
- ✓ Consenso temporal: 12 frames consecutivos válidos
- ✓ Auditoría completa en tabla `attendance` (score, margen, frames, timestamp)
- ✓ Anti-impostor: margen mínimo entre candidatos
- ✓ BD MySQL con referencia de foto y anti-duplicados
- ✓ Interfaz Tkinter completa
- ✓ `restore_db.py`: restaura BD desde dataset local

### 🔄 v3.0 (Próxima)
- [ ] Liveness detection (blink detection con MediaPipe)
- [ ] Exportar asistencia a CSV/Excel
- [ ] Cooldown efectivo: bloquear re-registro dentro de 5 min
- [ ] Interfaz web (Flask/Django)
- [ ] Logging centralizado de intentos fallidos

### 🎯 v4.0 (Mediano Plazo)
- [ ] Migración a FaceNet (99%+ precisión)
- [ ] API REST para integración con SIS
- [ ] Encriptación AES-256 de imágenes en BD
- [ ] Dashboard de estadísticas en tiempo real

### 🚀 v4.0 (Producción)
- [ ] Cloud deployment (Azure/AWS)
- [ ] Reconocimiento multi-modal
- [ ] Integración con SIS
- [ ] Mobile app
- [ ] On-device model optimization

## 📝 Licencia

Este proyecto es código abierto bajo licencia MIT. Úsalo libremente en proyectos personales o educativos.

## 👥 Contribuciones

Las contribuciones son bienvenidas:
1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Describir cambio'`)
4. Push (`git push origin feature/mejora`)
5. Abre un Pull Request

## 📞 Soporte

- **GitHub Issues:** Reporta bugs o solicita features
- **Email:** Contacta directamente al autor
- **Documentación:** Este README incluye troubleshooting completo

## 🙏 Agradecimientos

- **OpenCV community** por las herramientas de visión
- **mtcnn library** por la detección precisa
- **TensorFlow/Keras** por las redes neuronales
- **MySQL** por la persistencia de datos

---

⭐ **Si este proyecto te fue útil, considera:**
- Darle una estrella en GitHub
- Compartir con otros desarrolladores
- Reportar mejoras o nuevas ideas

