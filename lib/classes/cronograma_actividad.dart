import 'package:flutter/material.dart';

class CronogramaActividad {
  final int? id;
  final int cultivoId;
  final String titulo;
  final String descripcion;
  final String fechaProgramada;
  final String? fechaRealizada;
  final String
  tipo; // 'siembra', 'riego', 'fertilizacion', 'poda', 'cosecha', 'tratamiento', 'otro'
  final String estado; // 'pendiente', 'en_progreso', 'completada', 'cancelada'
  final double? costo;
  final String? notas;
  final String creadoEn;
  final String? actualizadoEn;

  CronogramaActividad({
    this.id,
    required this.cultivoId,
    required this.titulo,
    required this.descripcion,
    required this.fechaProgramada,
    this.fechaRealizada,
    required this.tipo,
    this.estado = 'pendiente',
    this.costo,
    this.notas,
    String? creadoEn,
    this.actualizadoEn,
  }) : creadoEn = creadoEn ?? DateTime.now().toIso8601String();

  // Tipos de actividades predefinidos
  static const List<String> TIPOS_ACTIVIDAD = [
    'siembra',
    'riego',
    'fertilizacion',
    'poda',
    'cosecha',
    'tratamiento',
    'monitoreo',
    'otro',
  ];

  static const List<String> ESTADOS_ACTIVIDAD = [
    'pendiente',
    'en_progreso',
    'completada',
    'cancelada',
  ];

  // Nombres amigables para los tipos
  static String getTipoNombre(String tipo) {
    switch (tipo) {
      case 'siembra':
        return 'Siembra';
      case 'riego':
        return 'Riego';
      case 'fertilizacion':
        return 'Fertilización';
      case 'poda':
        return 'Poda';
      case 'cosecha':
        return 'Cosecha';
      case 'tratamiento':
        return 'Tratamiento';
      case 'monitoreo':
        return 'Monitoreo';
      case 'otro':
        return 'Otro';
      default:
        return tipo;
    }
  }

  // Nombres amigables para los estados
  static String getEstadoNombre(String estado) {
    switch (estado) {
      case 'pendiente':
        return 'Pendiente';
      case 'en_progreso':
        return 'En Progreso';
      case 'completada':
        return 'Completada';
      case 'cancelada':
        return 'Cancelada';
      default:
        return estado;
    }
  }

  // Obtener icono según tipo
  static IconData getIconoTipo(String tipo) {
    switch (tipo) {
      case 'siembra':
        return Icons.agriculture;
      case 'riego':
        return Icons.water_drop;
      case 'fertilizacion':
        return Icons.eco;
      case 'poda':
        return Icons.content_cut;
      case 'cosecha':
        return Icons.grass;
      case 'tratamiento':
        return Icons.medication;
      case 'monitoreo':
        return Icons.search;
      case 'otro':
        return Icons.task_alt;
      default:
        return Icons.event;
    }
  }

  // Obtener color según estado
  static Color getColorEstado(String estado) {
    switch (estado) {
      case 'pendiente':
        return Colors.orange;
      case 'en_progreso':
        return Colors.blue;
      case 'completada':
        return Colors.green;
      case 'cancelada':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  // Convertir a Map para la base de datos
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'cultivoId': cultivoId,
      'titulo': titulo,
      'descripcion': descripcion,
      'fechaProgramada': fechaProgramada,
      'fechaRealizada': fechaRealizada,
      'tipo': tipo,
      'estado': estado,
      'costo': costo,
      'notas': notas,
      'creadoEn': creadoEn,
      'actualizadoEn': actualizadoEn ?? DateTime.now().toIso8601String(),
    };
  }

  // Crear desde Map de la base de datos
  factory CronogramaActividad.fromMap(Map<String, dynamic> map) {
    return CronogramaActividad(
      id: map['id']?.toInt(),
      cultivoId: map['cultivoId']?.toInt() ?? 0,
      titulo: map['titulo'] as String? ?? '',
      descripcion: map['descripcion'] as String? ?? '',
      fechaProgramada: map['fechaProgramada'] as String? ?? '',
      fechaRealizada: map['fechaRealizada'] as String?,
      tipo: map['tipo'] as String? ?? 'otro',
      estado: map['estado'] as String? ?? 'pendiente',
      costo: (map['costo'] is num) ? (map['costo'] as num).toDouble() : null,
      notas: map['notas'] as String?,
      creadoEn: map['creadoEn'] as String? ?? DateTime.now().toIso8601String(),
      actualizadoEn: map['actualizadoEn'] as String?,
    );
  }

  // Copiar con cambios
  CronogramaActividad copyWith({
    int? id,
    int? cultivoId,
    String? titulo,
    String? descripcion,
    String? fechaProgramada,
    String? fechaRealizada,
    String? tipo,
    String? estado,
    double? costo,
    String? notas,
    String? creadoEn,
    String? actualizadoEn,
  }) {
    return CronogramaActividad(
      id: id ?? this.id,
      cultivoId: cultivoId ?? this.cultivoId,
      titulo: titulo ?? this.titulo,
      descripcion: descripcion ?? this.descripcion,
      fechaProgramada: fechaProgramada ?? this.fechaProgramada,
      fechaRealizada: fechaRealizada ?? this.fechaRealizada,
      tipo: tipo ?? this.tipo,
      estado: estado ?? this.estado,
      costo: costo ?? this.costo,
      notas: notas ?? this.notas,
      creadoEn: creadoEn ?? this.creadoEn,
      actualizadoEn: actualizadoEn ?? this.actualizadoEn,
    );
  }

  @override
  String toString() {
    return 'CronogramaActividad(id: $id, cultivoId: $cultivoId, titulo: $titulo, tipo: $tipo, estado: $estado)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is CronogramaActividad &&
        other.id == id &&
        other.cultivoId == cultivoId &&
        other.titulo == titulo &&
        other.descripcion == descripcion &&
        other.fechaProgramada == fechaProgramada &&
        other.fechaRealizada == fechaRealizada &&
        other.tipo == tipo &&
        other.estado == estado &&
        other.costo == costo &&
        other.notas == notas;
  }

  @override
  int get hashCode {
    return Object.hash(
      id,
      cultivoId,
      titulo,
      descripcion,
      fechaProgramada,
      fechaRealizada,
      tipo,
      estado,
      costo,
      notas,
    );
  }
}
