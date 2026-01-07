// lib/classes/venta.dart
class Venta {
  final int? id;
  final int cultivoId;
  final String cultivoNombre;
  final double cantidad;
  final String unidad; // kg, toneladas, sacos, etc.
  final double precioUnitario;
  final double total;
  final String cliente;
  final String fecha; // formato: YYYY-MM-DD
  final String? notas;

  Venta({
    this.id,
    required this.cultivoId,
    required this.cultivoNombre,
    required this.cantidad,
    required this.unidad,
    required this.precioUnitario,
    required this.total,
    required this.cliente,
    required this.fecha,
    this.notas,
  });

  // Convertir de Map (BD) a objeto Venta
  factory Venta.fromMap(Map<String, dynamic> map) {
    return Venta(
      id: map['id'] as int?,
      cultivoId: map['cultivoId'] as int,
      cultivoNombre: map['cultivoNombre'] as String,
      cantidad: (map['cantidad'] as num).toDouble(),
      unidad: map['unidad'] as String,
      precioUnitario: (map['precioUnitario'] as num).toDouble(),
      total: (map['total'] as num).toDouble(),
      cliente: map['cliente'] as String,
      fecha: map['fecha'] as String,
      notas: map['notas'] as String?,
    );
  }

  // Convertir de objeto Venta a Map (para BD)
  Map<String, dynamic> toMap() {
    return {
      if (id != null) 'id': id,
      'cultivoId': cultivoId,
      'cultivoNombre': cultivoNombre,
      'cantidad': cantidad,
      'unidad': unidad,
      'precioUnitario': precioUnitario,
      'total': total,
      'cliente': cliente,
      'fecha': fecha,
      'notas': notas,
    };
  }

  // Copiar con modificaciones
  Venta copyWith({
    int? id,
    int? cultivoId,
    String? cultivoNombre,
    double? cantidad,
    String? unidad,
    double? precioUnitario,
    double? total,
    String? cliente,
    String? fecha,
    String? notas,
  }) {
    return Venta(
      id: id ?? this.id,
      cultivoId: cultivoId ?? this.cultivoId,
      cultivoNombre: cultivoNombre ?? this.cultivoNombre,
      cantidad: cantidad ?? this.cantidad,
      unidad: unidad ?? this.unidad,
      precioUnitario: precioUnitario ?? this.precioUnitario,
      total: total ?? this.total,
      cliente: cliente ?? this.cliente,
      fecha: fecha ?? this.fecha,
      notas: notas ?? this.notas,
    );
  }
}
