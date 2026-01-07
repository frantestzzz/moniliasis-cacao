// lib/Data/chat_database_helper.dart
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class ChatDatabaseHelper {
  static final ChatDatabaseHelper instance = ChatDatabaseHelper._();
  static Database? _database;

  ChatDatabaseHelper._();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'chat_history.db');

    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        // Tabla de conversaciones
        await db.execute('''
          CREATE TABLE conversaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            fechaCreacion TEXT NOT NULL,
            ultimaActualizacion TEXT NOT NULL
          )
        ''');

        // Tabla de mensajes
        await db.execute('''
          CREATE TABLE mensajes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversacionId INTEGER NOT NULL,
            tipo TEXT NOT NULL,
            contenido TEXT NOT NULL,
            esBot INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversacionId) REFERENCES conversaciones (id) ON DELETE CASCADE
          )
        ''');
      },
    );
  }

  // ========== CONVERSACIONES ==========

  // Crear nueva conversación
  Future<int> crearConversacion(String titulo) async {
    final db = await database;
    final now = DateTime.now().toIso8601String();

    return await db.insert('conversaciones', {
      'titulo': titulo,
      'fechaCreacion': now,
      'ultimaActualizacion': now,
    });
  }

  // Obtener todas las conversaciones
  Future<List<Map<String, dynamic>>> obtenerConversaciones() async {
    final db = await database;
    return await db.query(
      'conversaciones',
      orderBy: 'ultimaActualizacion DESC',
    );
  }

  // Actualizar última actualización de conversación
  Future<void> actualizarConversacion(int id) async {
    final db = await database;
    await db.update(
      'conversaciones',
      {'ultimaActualizacion': DateTime.now().toIso8601String()},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // Renombrar conversación
  Future<void> renombrarConversacion(int id, String nuevoTitulo) async {
    final db = await database;
    await db.update(
      'conversaciones',
      {'titulo': nuevoTitulo},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // Eliminar conversación
  Future<void> eliminarConversacion(int id) async {
    final db = await database;
    await db.delete('conversaciones', where: 'id = ?', whereArgs: [id]);
    // Los mensajes se eliminan automáticamente por el CASCADE
  }

  // ========== MENSAJES ==========

  // Guardar mensaje
  Future<int> guardarMensaje({
    required int conversacionId,
    required String tipo, // "text", "image", "document"
    required String contenido,
    required bool esBot,
  }) async {
    final db = await database;

    // Guardar mensaje
    final messageId = await db.insert('mensajes', {
      'conversacionId': conversacionId,
      'tipo': tipo,
      'contenido': contenido,
      'esBot': esBot ? 1 : 0,
      'timestamp': DateTime.now().toIso8601String(),
    });

    // Actualizar última actualización de la conversación
    await actualizarConversacion(conversacionId);

    return messageId;
  }

  // Obtener mensajes de una conversación
  Future<List<Map<String, dynamic>>> obtenerMensajes(int conversacionId) async {
    final db = await database;
    return await db.query(
      'mensajes',
      where: 'conversacionId = ?',
      whereArgs: [conversacionId],
      orderBy: 'timestamp ASC',
    );
  }

  // Contar mensajes de una conversación
  Future<int> contarMensajes(int conversacionId) async {
    final db = await database;
    final result = await db.rawQuery(
      'SELECT COUNT(*) as count FROM mensajes WHERE conversacionId = ?',
      [conversacionId],
    );
    return (result.first['count'] as int?) ?? 0;
  }

  // Obtener el primer mensaje de una conversación (para preview)
  Future<String?> obtenerPrimerMensaje(int conversacionId) async {
    final db = await database;
    final result = await db.query(
      'mensajes',
      where: 'conversacionId = ? AND esBot = 0',
      whereArgs: [conversacionId],
      orderBy: 'timestamp ASC',
      limit: 1,
    );

    if (result.isEmpty) return null;
    return result.first['contenido'] as String?;
  }

  // Limpiar base de datos (útil para testing)
  Future<void> limpiarTodo() async {
    final db = await database;
    await db.delete('mensajes');
    await db.delete('conversaciones');
  }
}
