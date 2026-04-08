"""
Restaura los registros de la tabla `user` en la BD
a partir de las imágenes existentes en la carpeta dataset/.
Ejecutar cuando la BD fue limpiada pero las imágenes locales siguen intactas.
"""

import os
import database as db

DATASET_PATH = "dataset/"


def restore():
    ok, info = db.testConnection()
    if not ok:
        print(f"[ERROR] No se pudo conectar a la BD: {info}")
        return

    print(f"[BD] Conectado: {info}\n")

    persons = sorted([
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, p))
    ])

    if not persons:
        print("[AVISO] No se encontraron carpetas en dataset/")
        return

    for person in persons:
        folder = os.path.join(DATASET_PATH, person)
        imgs = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        ])

        if not imgs:
            print(f"[SKIP]  '{person}' sin imágenes en la carpeta.")
            continue

        ref_path = os.path.join(folder, imgs[0])
        res = db.registerUser(person, ref_path)

        if res["affected"]:
            print(f"[OK]    '{person}' registrado en BD (id={res['id']})  —  ref: {imgs[0]}")
        else:
            print(f"[ERROR] No se pudo registrar '{person}' (¿ya existe en la BD?)")

    print("\nRestauración completada.")
    print("Ahora puedes entrenar el modelo desde la aplicación.")


if __name__ == "__main__":
    restore()
