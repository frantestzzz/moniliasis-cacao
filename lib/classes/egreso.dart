// lib/classes/egreso.dart

class Egreso {
  final int? id;
  final int cultivoId;
  final String cultivoNombre;
  final String tipo;
  final String descripcion;
  final double monto;
  final String fecha;
  final String? notas;

  Egreso({
    this.id,
    required this.cultivoId,
    required this.cultivoNombre,
    required this.tipo,
    required this.descripcion,
    required this.monto,
    required this.fecha,
    this.notas,
  });

  Map<String, Object?> toMap() {
    return {
      'id': id,
      'cultivoId': cultivoId, // Para el código Dart
      'cultivoNombre': cultivoNombre,
      'tipo': tipo,
      'descripcion': descripcion,
      'monto': monto,
      'fecha': fecha,
      'notas': notas,
    };
  }

  factory Egreso.fromMap(Map<String, dynamic> map) {
    return Egreso(
      id: map['id'] as int?,
      // ✅ Manejar ambos nombres de columna
      cultivoId: (map['cultivo_id'] ?? map['cultivoId']) as int,
      cultivoNombre: (map['cultivo_nombre'] ?? map['cultivoNombre']) as String,
      tipo: map['tipo'] as String,
      descripcion: map['descripcion'] as String,
      monto: (map['monto'] as num).toDouble(),
      fecha: map['fecha'] as String,
      notas: map['notas'] as String?,
    );
  }
}
