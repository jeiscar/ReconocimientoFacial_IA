"""
Script de diagnóstico: Prueba si la cámara funciona correctamente.
Ejecutar ANTES de abrir la app principal.
"""

import cv2
import sys

print("=" * 60)
print("DIAGNÓSTICO DE CÁMARA")
print("=" * 60)

# Probar abrir cámara
print("\n[1] Intentando abrir cámara (device 0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("    ✗ ERROR: No se pudo abrir la cámara.")
    print("\n    Soluciones:")
    print("    - Cierra Teams, Zoom, WhatsApp, OBS (apps que usan cámara)")
    print("    - Verifica que la cámara está conectada")
    print("    - Abre Administrador de dispositivos y busca 'Cámaras'")
    print("    - Intenta reiniciar la computadora")
    sys.exit(1)

print("    ✓ Cámara abierta correctamente")

# Configurar cámara
print("\n[2] Configurando parámetros de cámara...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("    ✓ Parámetros configurados")

# Probar lectura de frames
print("\n[3] Intentando leer 30 frames...")
frames_read = 0
frames_failed = 0

for i in range(30):
    ret, frame = cap.read()
    if ret:
        frames_read += 1
        if frames_read % 10 == 0:
            print(f"    ✓ {frames_read} frames leídos correctamente")
    else:
        frames_failed += 1
        print(f"    ✗ Fallo en frame {i+1}")

print(f"\n    Resultado: {frames_read}/30 frames leídos exitosamente")

if frames_read < 20:
    print("\n    ⚠️  ADVERTENCIA: La cámara está teniendo problemas.")
    print("       Intenta:")
    print("       - Cerrar otras apps que usan cámara")
    print("       - Reiniciar el dispositivo")
    print("       - Actualizar drivers de cámara")
else:
    print("\n    ✓ La cámara funciona correctamente")

# Mostrar propiedades
print("\n[4] Propiedades de la cámara:")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"    - Resolución: {int(width)} x {int(height)}")
print(f"    - FPS: {fps}")

cap.release()

print("\n" + "=" * 60)
if frames_read >= 20:
    print("✓ La cámara está LISTA para usar la aplicación")
else:
    print("✗ La cámara tiene PROBLEMAS. Intenta las soluciones arriba.")
print("=" * 60 + "\n")
