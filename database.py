import json
from pathlib import Path

import mysql.connector as db


BASE_DIR = Path(__file__).resolve().parent
KEYS_PATH = BASE_DIR / "keys.json"

with KEYS_PATH.open(encoding="utf-8") as json_file:
    keys = json.load(json_file)


def get_connection():
    return db.connect(**keys)


def testConnection():
    connection = None
    try:
        connection = get_connection()
        if connection.is_connected():
            server = connection.get_server_info()
            database_name = connection.database
            return True, f"Conectado a {database_name} en MySQL {server}"
        return False, "La conexion no quedo activa."
    except db.Error as error:
        return False, str(error)
    finally:
        if connection and connection.is_connected():
            connection.close()


def convert_to_binary_data(filename):
    try:
        with open(filename, "rb") as file:
            return file.read()
    except OSError:
        return None


def write_file(data, path):
    with open(path, "wb") as file:
        file.write(data)


def registerUser(name, photo):
    user_id = 0
    inserted = 0
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()
        sql = "INSERT INTO `user`(name, photo) VALUES (%s,%s)"
        picture = convert_to_binary_data(photo)

        if picture is not None:
            cursor.execute(sql, (name, picture))
            connection.commit()
            inserted = cursor.rowcount
            user_id = cursor.lastrowid
    except db.Error as error:
        print(f"Failed inserting image: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return {"id": user_id, "affected": inserted}


def getUser(name, path):
    user_id = 0
    rows = 0
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()
        sql = "SELECT * FROM `user` WHERE name = %s"

        cursor.execute(sql, (name,))
        records = cursor.fetchall()

        for row in records:
            user_id = row[0]
            write_file(row[2], path)
        rows = len(records)
    except db.Error as error:
        print(f"Failed to read image: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return {"id": user_id, "affected": rows}


def getAllUsers():
    users = []
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT idUser AS id, name FROM `user` ORDER BY idUser ASC")
        users = cursor.fetchall()
    except db.Error as error:
        print(f"Failed listing users: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return users


def deleteUser(name):
    deleted = 0
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM `user` WHERE name = %s", (name,))
        connection.commit()
        deleted = cursor.rowcount
    except db.Error as error:
        print(f"Failed deleting user: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return {"affected": deleted}


def recordAttendance(student_id, similarity_score, margin_score, frames_consensus, quality_check="OK"):
    """Registra asistencia verificada en tabla de auditoría."""
    attendance_id = 0
    inserted = 0
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()
        sql = """INSERT INTO attendance 
                 (student_id, similarity_score, margin_score, frames_consensus, quality_check, status)
                 VALUES (%s, %s, %s, %s, %s, 'VERIFIED')"""
        
        cursor.execute(sql, (student_id, similarity_score, margin_score, frames_consensus, quality_check))
        connection.commit()
        inserted = cursor.rowcount
        attendance_id = cursor.lastrowid
    except db.Error as error:
        print(f"Failed recording attendance: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return {"id": attendance_id, "affected": inserted}


def getAttendanceToday(student_id):
    """Obtiene registros de asistencia del día actual para un estudiante."""
    records = []
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        sql = """SELECT * FROM attendance 
                 WHERE student_id = %s AND DATE(timestamp) = CURDATE()
                 ORDER BY timestamp DESC"""
        
        cursor.execute(sql, (student_id,))
        records = cursor.fetchall()
    except db.Error as error:
        print(f"Failed retrieving attendance: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return records


def test_connection():
    return testConnection()


def register_user(name, photo):
    return registerUser(name, photo)


def get_user(name, path):
    return getUser(name, path)


def get_all_users():
    return getAllUsers()


def delete_user(name):
    return deleteUser(name)


def record_attendance(student_id, similarity_score, margin_score, frames_consensus, quality_check="OK"):
    return recordAttendance(student_id, similarity_score, margin_score, frames_consensus, quality_check)