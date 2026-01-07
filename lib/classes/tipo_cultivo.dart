class TipoCultivo {
  final int? id;
  final String nombre;

  TipoCultivo({this.id, required this.nombre});

  factory TipoCultivo.fromMap(Map<String, Object?> map) => TipoCultivo(
    id: map['id'] as int?,
    nombre: map['nombre'] as String? ?? '',
  );

  Map<String, Object?> toMap() => {if (id != null) 'id': id, 'nombre': nombre};
}
