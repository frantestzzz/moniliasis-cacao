// lib/classes/cultivo.dart
import 'dart:convert';

class Cultivo {
  final int? id;
  final String nombre;
  final String tipoSuelo;
  final double area;
  final String fechaSiembra;
  final String? fechaCosecha;
  String estado; // "activo", "inactivo", "cosechado", "en_riesgo"
  final String? notas;
  final String? imagenUrl; // Ruta local de la imagen
  final int? tipoId;
  final int? categoriaId;
  final String tipoRiego;
  final double? cantidadCosechada;
  final double? ingresos;
  final double? egresos;
  String? razonRiesgo;
  String? tipoRiesgo;
  String? fechaRiesgo;
  String? fechaInicioRiesgo;
  String? nivelGravedad;
  String? fechaFinRiesgo; // Fecha cuando terminó el riesgo
  String? historialRiesgos; // JSON con historial de riesgos anteriores

  Cultivo({
    this.id,
    required this.nombre,
    required this.tipoSuelo,
    required this.area,
    required this.fechaSiembra,
    this.fechaCosecha,
    required this.estado,
    this.notas,
    this.imagenUrl,
    this.tipoId,
    this.categoriaId,
    required this.tipoRiego,
    this.cantidadCosechada,
    this.ingresos,
    this.egresos,
    this.razonRiesgo,
    this.tipoRiesgo,
    this.fechaRiesgo,
    this.fechaInicioRiesgo,
    this.nivelGravedad,
    this.fechaFinRiesgo,
    this.historialRiesgos,
  });

  // Estados válidos
  static const String ESTADO_ACTIVO = 'activo';
  static const String ESTADO_INACTIVO = 'inactivo';
  static const String ESTADO_COSECHADO = 'cosechado';
  static const String ESTADO_EN_RIESGO = 'en_riesgo';

  bool get esActivo => estado == ESTADO_ACTIVO;
  bool get esInactivo => estado == ESTADO_INACTIVO;
  bool get esCosechado => estado == ESTADO_COSECHADO;
  bool get esEnRiesgo => estado == ESTADO_EN_RIESGO;

  factory Cultivo.fromMap(Map<String, dynamic> map) {
    return Cultivo(
      id: map['id'] as int?,
      nombre: map['nombre'] as String? ?? '',
      tipoSuelo: map['tipoSuelo'] as String? ?? '',
      area: (map['area'] is num) ? (map['area'] as num).toDouble() : 0.0,
      fechaSiembra: map['fechaSiembra'] as String? ?? '',
      fechaCosecha: map['fechaCosecha'] as String?,
      estado: (map['estado'] as String? ?? 'activo').toLowerCase(),
      notas: map['notas'] as String?,
      imagenUrl: map['imagenUrl'] as String?,
      tipoId: map['tipoId'] as int?,
      categoriaId: map['categoriaId'] as int?,
      tipoRiego: map['tipoRiego'] as String? ?? '',
      cantidadCosechada: map['cantidadCosechada'] != null
          ? (map['cantidadCosechada'] as num).toDouble()
          : null,
      ingresos: map['ingresos'] != null
          ? (map['ingresos'] as num).toDouble()
          : null,
      egresos: map['egresos'] != null
          ? (map['egresos'] as num).toDouble()
          : null,
      razonRiesgo: map['riskReason'] as String?,
      tipoRiesgo: map['riskType'] as String?,
      fechaRiesgo: map['riskDate'] as String?,
      fechaInicioRiesgo: map['riskStartDate'] as String?,
      nivelGravedad: map['riskSeverity'] as String?,
      fechaFinRiesgo: map['riskEndDate'] as String?,
      historialRiesgos: map['riskHistory'] as String?,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'nombre': nombre,
      'tipoSuelo': tipoSuelo,
      'area': area,
      'fechaSiembra': fechaSiembra,
      'fechaCosecha': fechaCosecha,
      'estado': estado.toLowerCase(),
      'notas': notas,
      'imagenUrl': imagenUrl,
      'tipoId': tipoId,
      'categoriaId': categoriaId,
      'tipoRiego': tipoRiego,
      'cantidadCosechada': cantidadCosechada,
      'ingresos': ingresos,
      'egresos': egresos,
      'riskReason': razonRiesgo,
      'riskType': tipoRiesgo,
      'riskDate': fechaRiesgo,
      'riskStartDate': fechaInicioRiesgo,
      'riskSeverity': nivelGravedad,
      'riskEndDate': fechaFinRiesgo,
      'riskHistory': historialRiesgos,
      // Mantener compatibilidad con el campo isRisk
      'isRisk': estado == ESTADO_EN_RIESGO ? 1 : 0,
    };
  }
}

// Extension para manejar historial de riesgos
extension CultivoRiesgoExtension on Cultivo {
  // 🆕 CORREGIDO: Agregar riesgo actual al historial con fecha de fin en formato DD/MM/YYYY
  Map<String, dynamic> agregarRiesgoAlHistorial() {
    if (razonRiesgo == null || tipoRiesgo == null) return toMap();

    // 🆕 Formatear fecha de fin en DD/MM/YYYY
    final now = DateTime.now();
    final fechaFinFormateada =
        '${now.day.toString().padLeft(2, '0')}/${now.month.toString().padLeft(2, '0')}/${now.year}';

    final riesgoActual = {
      'razon': razonRiesgo,
      'tipo': tipoRiesgo,
      'fechaRegistro': fechaRiesgo,
      'fechaInicio': fechaInicioRiesgo,
      'nivelGravedad': nivelGravedad,
      'fechaFin': fechaFinFormateada, // 🆕 Formato DD/MM/YYYY
    };

    List<Map<String, dynamic>> historial = [];
    if (historialRiesgos != null && historialRiesgos!.isNotEmpty) {
      try {
        final decoded = json.decode(historialRiesgos!) as List;
        historial = decoded.cast<Map<String, dynamic>>();
      } catch (e) {
        // Si hay error, empezamos con historial vacío
      }
    }

    // 🆕 AGREGAR a la lista existente, no reemplazar
    historial.add(riesgoActual);

    // 🆕 DEBUG: Imprimir para verificar
    print('Historial después de agregar: ${json.encode(historial)}');

    return Cultivo(
      id: id,
      nombre: nombre,
      tipoSuelo: tipoSuelo,
      area: area,
      fechaSiembra: fechaSiembra,
      fechaCosecha: fechaCosecha,
      estado: estado,
      notas: notas,
      imagenUrl: imagenUrl,
      tipoId: tipoId,
      categoriaId: categoriaId,
      tipoRiego: tipoRiego,
      cantidadCosechada: cantidadCosechada,
      ingresos: ingresos,
      egresos: egresos,
      // Limpiar campos de riesgo actual
      razonRiesgo: null,
      tipoRiesgo: null,
      fechaRiesgo: null,
      fechaInicioRiesgo: null,
      nivelGravedad: null,
      fechaFinRiesgo: null,
      // Guardar historial completo
      historialRiesgos: json.encode(historial),
    ).toMap();
  }

  // Obtener lista de riesgos históricos
  List<Map<String, dynamic>> getHistorialRiesgos() {
    if (historialRiesgos == null || historialRiesgos!.isEmpty) return [];

    try {
      final decoded = json.decode(historialRiesgos!) as List;
      final historial = decoded.cast<Map<String, dynamic>>();

      // 🆕 DEBUG: Imprimir para verificar lectura
      print('getHistorialRiesgos - Leídos ${historial.length} riesgos');
      print('getHistorialRiesgos - Datos: ${json.encode(historial)}');

      return historial;
    } catch (e) {
      print('Error en getHistorialRiesgos: $e');
      return [];
    }
  }

  // 🆕 Obtener conteo total de veces en riesgo (incluyendo la actual si está en riesgo)
  int get vecesEnRiesgo {
    final historial = getHistorialRiesgos();
    int conteo = historial.length;

    // Si está actualmente en riesgo, sumar uno más
    if (esEnRiesgo && razonRiesgo != null) {
      conteo++;
    }

    // 🆕 DEBUG: Imprimir conteo final
    print(
      'vecesEnRiesgo - Historial: ${historial.length}, Actual: ${esEnRiesgo ? 1 : 0}, Total: $conteo',
    );

    return conteo;
  }

  // Verificar si tiene historial de riesgos
  bool get tieneHistorialRiesgos {
    return historialRiesgos != null && historialRiesgos!.isNotEmpty;
  }
}
