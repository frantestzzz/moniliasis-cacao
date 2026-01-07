// lib/Model/configuracion.dart
class Configuracion {
  final int? id;
  final String tema; // 'light', 'dark', 'auto'
  final String tamanoFuente; // 'small', 'medium', 'large'
  final bool animaciones;
  final bool vibracion;

  Configuracion({
    this.id,
    this.tema = 'light',
    this.tamanoFuente = 'medium',
    this.animaciones = true,
    this.vibracion = true,
  });

  factory Configuracion.fromMap(Map<String, dynamic> map) {
    return Configuracion(
      id: map['id'] as int?,
      tema: map['tema'] as String? ?? 'light',
      tamanoFuente: map['tamanoFuente'] as String? ?? 'medium',
      animaciones: (map['animaciones'] as int? ?? 1) == 1,
      vibracion: (map['vibracion'] as int? ?? 1) == 1,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'tema': tema,
      'tamanoFuente': tamanoFuente,
      'animaciones': animaciones ? 1 : 0,
      'vibracion': vibracion ? 1 : 0,
    };
  }

  Configuracion copyWith({
    int? id,
    String? tema,
    String? tamanoFuente,
    bool? animaciones,
    bool? vibracion,
  }) {
    return Configuracion(
      id: id ?? this.id,
      tema: tema ?? this.tema,
      tamanoFuente: tamanoFuente ?? this.tamanoFuente,
      animaciones: animaciones ?? this.animaciones,
      vibracion: vibracion ?? this.vibracion,
    );
  }
}
