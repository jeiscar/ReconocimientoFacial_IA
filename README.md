# Sistema de Reconocimiento Facial para Control de Asistencia

Sistema de registro y autenticación de usuarios mediante reconocimiento facial, diseñado para la toma de asistencia de estudiantes. Utiliza inteligencia artificial para detectar y comparar rostros con alta precisión.

## 📋 Descripción

Aplicativo de escritorio que permite:
- **Registro de usuarios** mediante captura facial con webcam
- **Autenticación biométrica** por reconocimiento facial
- **Almacenamiento seguro** en base de datos MySQL
- **Interfaz gráfica intuitiva** con Tkinter

## ✨ Características

- Detección de rostros en tiempo real con MTCNN
- Comparación facial usando descriptores ORB
- Umbral de similitud del 94% para autenticación
- Almacenamiento de imágenes en formato BLOB
- Interfaz gráfica minimalista y funcional
- Manejo de errores y validación de datos

## 🛠️ Tecnologías y Librerías

| Librería | Versión | Propósito |
|----------|---------|-----------|
| **Python** | 3.11+ | Lenguaje base |
| **Tkinter** | Built-in | Interfaz gráfica |
| **OpenCV** | 4.x | Procesamiento de imágenes y comparación facial |
| **Matplotlib** | 3.x | Manipulación y visualización de imágenes |
| **MTCNN** | 0.1.1 | Detección precisa de rostros |
| **TensorFlow** | 2.x | Backend para MTCNN |
| **MySQL Connector** | 8.x | Conexión con base de datos |

### ¿Por qué estas librerías?

- **OpenCV:** Herramientas robustas para visión por computadora, incluye algoritmos ORB y BFMatcher
- **MTCNN:** Red neuronal especializada en detección facial con alta precisión
- **Tkinter:** Solución nativa de Python para interfaces gráficas multiplataforma
- **MySQL:** Base de datos relacional para almacenamiento persistente con soporte BLOB

## 📦 Requisitos Previos

- Python 3.11 o superior
- MySQL Server 5.7+ o MariaDB
- Webcam funcional
- Windows/Linux/macOS

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/jeiscar/ReconocimientoFacial_IA.git
cd ReconocimientoFacial_IA
```

### 2. Crear entorno virtual (recomendado)
```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install opencv-python matplotlib mtcnn mysql-connector-python tensorflow
```

### 4. Configurar base de datos

**a) Crear la base de datos:**
```bash
mysql -u root -p < script.sql
```

O ejecutar manualmente en MySQL:
```sql
CREATE DATABASE agency;
USE agency;

CREATE TABLE user(
 idUser INT AUTO_INCREMENT,
 name VARCHAR(75) NOT NULL,
 photo LONGBLOB,
 CONSTRAINT pk_user_idUser PRIMARY KEY(idUser)
);
```

**b) Configurar credenciales:**

Editar `keys.json` con tus credenciales de MySQL:
```json
{
    "host": "localhost",
    "user": "root",
    "password": "tu_password",
    "database": "agency"
}
```

### 5. Verificar configuración de rutas

En `facial_recognition.py`, ajusta la ruta si es necesario:
```python
path = "C:/facial_recognition-main/img"  # Tu ruta absoluta
```

## ▶️ Uso

### Ejecutar la aplicación
```bash
python facial_recognition.py
```

### Flujo de trabajo

#### Registrar nuevo usuario:
1. Clic en "Registrarse"
2. Ingresar nombre del estudiante
3. Clic en "Capturar rostro"
4. Presionar **ESC** para tomar la foto
5. El sistema detectará y guardará el rostro

#### Tomar asistencia (Login):
1. Clic en "Toma asistencia"
2. Ingresar nombre del estudiante
3. Clic en "Capturar rostro"
4. Presionar **ESC** para tomar la foto
5. El sistema comparará y validará el rostro
6. Mensaje de éxito si la similitud ≥ 94%

## 📁 Estructura del Proyecto

```
ReconocimientoFacial_IA/
│
├── facial_recognition.py  # Aplicación principal (GUI + lógica)
├── database.py            # Funciones de conexión y operaciones DB
├── script.sql             # Script de creación de base de datos
├── keys.json              # Credenciales MySQL (no versionado)
├── README.md              # Documentación
├── .gitignore             # Archivos excluidos de Git
│
├── img/                   # Carpeta para imágenes temporales
└── __pycache__/           # Cache de Python (no versionado)
```

## 🔬 Funcionamiento Técnico

### 1. Proceso de Registro

```
Usuario ingresa nombre → Captura webcam → Detección facial (MTCNN) → 
Recorte y redimensión (150x200px) → Conversión a binario → 
Almacenamiento en MySQL
```

**Código clave:**
```python
def register_capture():
    cap = cv2.VideoCapture(0)  # Iniciar webcam
    user_reg_img = user1.get()
    img = f"{user_reg_img}.jpg"
    
    # Captura en loop hasta ESC
    while True:
        ret, frame = cap.read()
        cv2.imshow("Registro Facial", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cv2.imwrite(img, frame)
    cap.release()
    
    # Detección y recorte facial
    pixels = plt.imread(img)
    faces = MTCNN().detect_faces(pixels)
    face(img, faces)  # Recortar rostro
    register_face_db(img)  # Guardar en DB
```

### 2. Proceso de Autenticación

```
Usuario ingresa nombre → Captura webcam → Detección facial → 
Recuperar imagen de DB → Comparación ORB + BFMatcher → 
Cálculo de similitud → Validación (≥94%) → Autenticación
```

**Algoritmo de comparación:**
```python
def compatibility(img1, img2):
    orb = cv2.ORB_create()
    
    # Extraer características
    kpa, dac1 = orb.detectAndCompute(img1, None)
    kpa, dac2 = orb.detectAndCompute(img2, None)
    
    # Comparar descriptores
    comp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = comp.match(dac1, dac2)
    
    # Filtrar coincidencias (distancia < 70)
    similar = [x for x in matches if x.distance < 70]
    
    if len(matches) == 0:
        return 0
    return len(similar)/len(matches)  # Ratio de similitud
```

### 3. Detección Facial con MTCNN

MTCNN (Multi-task Cascaded Convolutional Networks):
- Detecta rostros en 3 etapas (P-Net, R-Net, O-Net)
- Proporciona coordenadas del bounding box
- Identifica puntos faciales clave (ojos, nariz, boca)

```python
def face(img, faces):
    data = plt.imread(img)
    for i in range(len(faces)):
        x1, y1, ancho, alto = faces[i]["box"]
        x2, y2 = x1 + ancho, y1 + alto
        
        # Recortar y redimensionar rostro
        face = cv2.resize(data[y1:y2, x1:x2], (150,200), 
                         interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img, face)
```

## 🧠 Tipo de Solución ML

**Clasificación: Proyecto con Machine Learning (predictivo)**

- **Detección de objetos:** MTCNN para localizar rostros
- **Clasificación binaria:** Verificación de identidad (Sí/No es el usuario)
- **Comparación de características:** ORB descriptors + matching

## 📊 Dataset

### Características
- **Tipo:** Dataset propio generado por captura en tiempo real
- **Formato:** Imágenes JPG (150x200 píxeles)
- **Almacenamiento:** LONGBLOB en MySQL
- **Contenido:** 
  - ID de usuario (auto-incremental)
  - Nombre del estudiante
  - Fotografía facial (binario)
  - Timestamp implícito

### Alternativas para validación
- **LFW (Labeled Faces in the Wild):** 13,000+ imágenes de rostros
- **CASIA-WebFace:** 500,000 imágenes para reconocimiento facial

### Riesgos Identificados

| Riesgo | Descripción | Mitigación |
|--------|-------------|-----------|
| **Sesgo facial** | Precisión variable según etnia/género | Diversificar muestras de entrenamiento |
| **Datos personales** | Cumplimiento GDPR/protección de datos | Encriptación, consentimiento informado |
| **Calidad de imagen** | Iluminación pobre, ángulos incorrectos | Validación previa, guías visuales |
| **Variabilidad temporal** | Cambios en apariencia (barba, gafas) | Múltiples registros por usuario |
| **Datos faltantes** | Usuario sin registro, fallos de captura | Manejo de excepciones robusto |

## 🎯 Algoritmos Candidatos

### 1. **ORB + BFMatcher** (Implementado ✓)
- **Ventajas:** Rápido, sin GPU, robusto a rotaciones
- **Desventajas:** Sensible a iluminación
- **Precisión:** ~85-92%

### 2. **FaceNet + Embeddings** (Recomendado para producción)
- **Ventajas:** Alta precisión (99%), estándar industrial
- **Desventajas:** Requiere GPU para inferencia rápida
- **Implementación:**
```python
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2')
embedding_user = model(face_image_user)
embedding_login = model(face_image_login)

distance = np.linalg.norm(embedding_user - embedding_login)
if distance < 0.6:
    authenticated = True
```

### 3. **VGGFace2 + One-Class SVM**
- **Ventajas:** Detecta anomalías, funciona con pocas muestras
- **Desventajas:** Requiere entrenamiento por usuario

### 4. **DeepFace (Librería simplificada)**
- **Ventajas:** Implementación sencilla, múltiples modelos
- **Desventajas:** No optimizado para producción

### 5. **Autoencoder + Detección de Anomalías**
- **Ventajas:** Aprendizaje no supervisado
- **Desventajas:** Requiere GPU y datos de entrenamiento

## ⚙️ Configuración Avanzada

### Ajustar umbral de similitud
En `facial_recognition.py`, línea ~167:
```python
if comp >= 0.94:  # Cambiar umbral aquí (0.0 - 1.0)
    printAndShow(screen2, f"Bienvenido, {user_login}", 1)
```

### Cambiar resolución de rostros
En función `face()`, línea ~70:
```python
face = cv2.resize(data[y1:y2, x1:x2], (150,200), ...)  # (ancho, alto)
```

## 🐛 Solución de Problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```
**Nota:** TensorFlow requiere Python ≤ 3.11

### Error: "Can't connect to MySQL server"
- Verificar que MySQL esté corriendo
- Comprobar credenciales en `keys.json`
- Verificar firewall/puertos (3306)

### Error: "No faces detected"
- Mejorar iluminación
- Encuadrar rostro correctamente en la cámara
- Verificar que la webcam funcione

### La comparación siempre falla
- Verificar que la ruta en `path` sea correcta
- Comprobar que la carpeta `img/` exista
- Ajustar umbral de similitud si es necesario

## 📈 Roadmap de Mejoras

- [ ] Migrar a FaceNet para mayor precisión
- [ ] Implementar detección de vida (liveness detection)
- [ ] Agregar logging de intentos de acceso
- [ ] Exportar reportes de asistencia a CSV/Excel
- [ ] Interfaz web con Flask/Django
- [ ] Reconocimiento multi-facial simultáneo
- [ ] Encriptación de imágenes en base de datos
- [ ] Soporte para múltiples cámaras
- [ ] Dashboard de estadísticas

## 📝 Licencia

Este proyecto es de código abierto para fines educativos.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

## 📧 Contacto

**Autor:** jeiscar  
**Repositorio:** [github.com/jeiscar/ReconocimientoFacial_IA](https://github.com/jeiscar/ReconocimientoFacial_IA)

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub