CREATE DATABASE IF NOT EXISTS agency CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE agency;

CREATE TABLE IF NOT EXISTS user(
    idUser INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    photo LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla de auditoría para registro de asistencia
CREATE TABLE IF NOT EXISTS attendance(
    idAttendance INT AUTO_INCREMENT PRIMARY KEY,
    idUser INT NOT NULL,
    student_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    similarity_score FLOAT NOT NULL,
    margin_score FLOAT NOT NULL,
    status ENUM('VERIFIED', 'REJECTED', 'QUALITY_ERROR') DEFAULT 'VERIFIED',
    frames_consensus INT NOT NULL,
    quality_check VARCHAR(50),
    FOREIGN KEY (idUser) REFERENCES user(idUser) ON DELETE CASCADE,
    INDEX idx_student_id (student_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SELECT COUNT(*) as total_users FROM user;
SELECT COUNT(*) as total_attendance FROM attendance;