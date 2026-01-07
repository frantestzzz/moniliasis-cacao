import 'dart:convert';
import 'package:crypto/crypto.dart';

class Usuario {
  final int? id;
  final String nombre;
  final String correo;
  final String passwordHash;

  Usuario({
    this.id,
    required this.nombre,
    required this.correo,
    required this.passwordHash,
  });

  // Convertir un mapa a objeto Usuario
  factory Usuario.fromMap(Map<String, dynamic> map) {
    return Usuario(
      id: map['id'] as int?,
      nombre: map['nombre'] as String,
      correo: map['correo'] as String,
      passwordHash: map['passwordHash'] as String,
    );
  }

  // Convertir objeto Usuario a mapa
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'nombre': nombre,
      'correo': correo,
      'passwordHash': passwordHash,
    };
  }

  // Generar hash SHA-256 de la contraseña
  static String hashPassword(String password) {
    return sha256.convert(utf8.encode(password)).toString();
  }

  // Verificar si la contraseña coincide con el hash
  bool verifyPassword(String password) {
    return hashPassword(password) == passwordHash;
  }
}
