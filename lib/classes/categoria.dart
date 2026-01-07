class Categoria {
  final int? id;
  final String nombre;

  Categoria({this.id, required this.nombre});

  factory Categoria.fromMap(Map<String, Object?> map) =>
      Categoria(id: map['id'] as int?, nombre: map['nombre'] as String? ?? '');

  Map<String, Object?> toMap() => {if (id != null) 'id': id, 'nombre': nombre};
}
