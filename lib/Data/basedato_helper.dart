// lib/Data/basedato_helper.dart
import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

class BasedatoHelper {
  BasedatoHelper._privateConstructor();
  static final BasedatoHelper instance = BasedatoHelper._privateConstructor();

  Database? _database;

  Future<Database> openDataBase() async {
    if (_database != null) return _database!;

    final databasePath = await getDatabasesPath();
    final path = join(databasePath, 'mydatabase.db');

    _database = await openDatabase(
      path,
      version: 8,
      onCreate: (db, version) async {
        await _createAllTables(db);
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('''
            CREATE TABLE IF NOT EXISTS ventas (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              cultivoId INTEGER NOT NULL,
              cultivoNombre TEXT NOT NULL,
              cantidad REAL NOT NULL,
              unidad TEXT NOT NULL,
              precioUnitario REAL NOT NULL,
              total REAL NOT NULL,
              cliente TEXT NOT NULL,
              fecha TEXT NOT NULL,
              notas TEXT,
              FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE CASCADE
            )
          ''');
        }

        if (oldVersion < 8) {
          // Verificar si la columna imagenPerfil ya existe antes de agregarla
          final result = await db.rawQuery("PRAGMA table_info(usuarios)");
          final hasImagenPerfil = result.any(
            (column) => column['name'] == 'imagenPerfil',
          );

          if (!hasImagenPerfil) {
            await db.execute(
              'ALTER TABLE usuarios ADD COLUMN imagenPerfil TEXT',
            );
          }
        }

        if (oldVersion < 3) {
          await db.execute('''
            CREATE TABLE IF NOT EXISTS conversaciones (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              titulo TEXT NOT NULL,
              fechaCreacion TEXT NOT NULL,
              ultimaActualizacion TEXT NOT NULL,
              mensajesCount INTEGER DEFAULT 0
            )
          ''');

          await db.execute('''
            CREATE TABLE IF NOT EXISTS mensajes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              conversacionId INTEGER NOT NULL,
              tipo TEXT NOT NULL,
              contenido TEXT NOT NULL,
              fecha TEXT NOT NULL,
              archivoPath TEXT,
              archivoNombre TEXT,
              archivoTipo TEXT,
              FOREIGN KEY (conversacionId) REFERENCES conversaciones (id) ON DELETE CASCADE
            )
          ''');
        }

        if (oldVersion < 4) {
          // 🆕 Crear tablas de alertas y tareas
          await db.execute('''
            CREATE TABLE IF NOT EXISTS alertas (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              cultivoId INTEGER,
              tipo TEXT NOT NULL,
              severidad TEXT NOT NULL,
              titulo TEXT NOT NULL,
              mensaje TEXT NOT NULL,
              fecha TEXT NOT NULL,
              resuelta INTEGER DEFAULT 0,
              rutaDestino TEXT,
              FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE CASCADE
            )
          ''');

          await db.execute('''
            CREATE TABLE IF NOT EXISTS tareas (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              cultivoId INTEGER,
              titulo TEXT NOT NULL,
              descripcion TEXT,
              categoria TEXT NOT NULL,
              fechaProgramada TEXT NOT NULL,
              completada INTEGER DEFAULT 0,
              prioridad TEXT DEFAULT 'media',
              FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE SET NULL
            )
          ''');
        }

        if (oldVersion < 5) {
          // 🆕 Crear tabla de cronograma de actividades
          await db.execute('''
            CREATE TABLE IF NOT EXISTS cronograma_actividades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              cultivoId INTEGER NOT NULL,
              titulo TEXT NOT NULL,
              descripcion TEXT NOT NULL,
              fechaProgramada TEXT NOT NULL,
              fechaRealizada TEXT,
              tipo TEXT NOT NULL,
              estado TEXT NOT NULL DEFAULT 'pendiente',
              costo REAL,
              notas TEXT,
              creadoEn TEXT NOT NULL,
              actualizadoEn TEXT,
              FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE CASCADE
            )
          ''');
        }
      },
      onOpen: (db) async {
        await _createAllTables(db);
        await _addMissingColumns(db);
      },
    );

    return _database!;
  }

  Future<void> _createAllTables(Database db) async {
    await db.execute(
      'CREATE TABLE IF NOT EXISTS mitabla (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)',
    );

    await db.execute(
      'CREATE TABLE IF NOT EXISTS tipos_cultivo (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT NOT NULL)',
    );

    await db.execute(
      'CREATE TABLE IF NOT EXISTS categorias (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT NOT NULL)',
    );

    await db.execute(
      'CREATE TABLE IF NOT EXISTS cultivos (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT NOT NULL, tipoSuelo TEXT NOT NULL, area REAL NOT NULL, fechaSiembra TEXT NOT NULL, fechaCosecha TEXT, estado TEXT NOT NULL, notas TEXT, imagenUrl TEXT, tipoId INTEGER, categoriaId INTEGER, tipoRiego TEXT, cantidadCosechada REAL, ingresos REAL, egresos REAL, isRisk INTEGER DEFAULT 0, riskReason TEXT, riskType TEXT, riskDate TEXT, riskStartDate TEXT, riskSeverity TEXT, riskEndDate TEXT, riskHistory TEXT)',
    );

    await db.execute('''
      CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        correo TEXT UNIQUE NOT NULL,
        passwordHash TEXT NOT NULL,
        resetToken TEXT,
        resetTokenExpiry INTEGER,
        imagenPerfil TEXT
      )
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS ventas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cultivoId INTEGER NOT NULL,
        cultivoNombre TEXT NOT NULL,
        cantidad REAL NOT NULL,
        unidad TEXT NOT NULL,
        precioUnitario REAL NOT NULL,
        total REAL NOT NULL,
        cliente TEXT NOT NULL,
        fecha TEXT NOT NULL,
        notas TEXT,
        FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE CASCADE
      )
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS conversaciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        titulo TEXT NOT NULL,
        fechaCreacion TEXT NOT NULL,
        ultimaActualizacion TEXT NOT NULL,
        mensajesCount INTEGER DEFAULT 0
      )
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS mensajes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversacionId INTEGER NOT NULL,
        tipo TEXT NOT NULL,
        contenido TEXT NOT NULL,
        fecha TEXT NOT NULL,
        archivoPath TEXT,
        archivoNombre TEXT,
        archivoTipo TEXT,
        FOREIGN KEY (conversacionId) REFERENCES conversaciones (id) ON DELETE CASCADE
      )
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS cronograma_actividades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cultivoId INTEGER NOT NULL,
        titulo TEXT NOT NULL,
        descripcion TEXT NOT NULL,
        fechaProgramada TEXT NOT NULL,
        fechaRealizada TEXT,
        tipo TEXT NOT NULL,
        estado TEXT NOT NULL DEFAULT 'pendiente',
        costo REAL,
        notas TEXT,
        creadoEn TEXT NOT NULL,
        actualizadoEn TEXT,
        FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE CASCADE
      )
    ''');

    // 🆕 TABLAS DE ALERTAS Y TAREAS
    await db.execute('''
      CREATE TABLE IF NOT EXISTS alertas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cultivoId INTEGER,
        tipo TEXT NOT NULL,
        severidad TEXT NOT NULL,
        titulo TEXT NOT NULL,
        mensaje TEXT NOT NULL,
        fecha TEXT NOT NULL,
        resuelta INTEGER DEFAULT 0,
        rutaDestino TEXT,
        FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE CASCADE
      )
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS tareas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cultivoId INTEGER,
        titulo TEXT NOT NULL,
        descripcion TEXT,
        categoria TEXT NOT NULL,
        fechaProgramada TEXT NOT NULL,
        completada INTEGER DEFAULT 0,
        prioridad TEXT DEFAULT 'media',
        FOREIGN KEY (cultivoId) REFERENCES cultivos (id) ON DELETE SET NULL
      )
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS egresos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cultivo_id INTEGER NOT NULL,
        cultivo_nombre TEXT NOT NULL,
        tipo TEXT NOT NULL,
        descripcion TEXT NOT NULL,
        monto REAL NOT NULL,
        fecha TEXT NOT NULL,
        notas TEXT,
        FOREIGN KEY (cultivo_id) REFERENCES cultivos (id) ON DELETE CASCADE
      )
    ''');
  }

  Future<void> _addMissingColumns(Database db) async {
    final columns = await db.rawQuery('PRAGMA table_info(cultivos)');
    final columnNames = columns.map((c) => c['name'] as String).toList();

    if (!columnNames.contains('isRisk')) {
      await db.execute(
        'ALTER TABLE cultivos ADD COLUMN isRisk INTEGER DEFAULT 0',
      );
    }
    if (!columnNames.contains('riskReason')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskReason TEXT');
    }
    if (!columnNames.contains('riskType')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskType TEXT');
    }
    if (!columnNames.contains('riskDate')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskDate TEXT');
    }
    if (!columnNames.contains('cantidadCosechada')) {
      await db.execute(
        'ALTER TABLE cultivos ADD COLUMN cantidadCosechada REAL',
      );
    }
    if (!columnNames.contains('ingresos')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN ingresos REAL');
    }
    if (!columnNames.contains('egresos')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN egresos REAL');
    }

    // Nuevas columnas para historial de riesgos
    if (!columnNames.contains('riskStartDate')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskStartDate TEXT');
    }
    if (!columnNames.contains('riskSeverity')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskSeverity TEXT');
    }
    if (!columnNames.contains('riskEndDate')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskEndDate TEXT');
    }
    if (!columnNames.contains('riskHistory')) {
      await db.execute('ALTER TABLE cultivos ADD COLUMN riskHistory TEXT');
    }

    // Verificar y crear tabla egresos si no existe
    final tables = await db.rawQuery(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='egresos'",
    );
    if (tables.isEmpty) {
      await db.execute('''
        CREATE TABLE IF NOT EXISTS egresos (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          cultivo_id INTEGER NOT NULL,
          cultivo_nombre TEXT NOT NULL,
          tipo TEXT NOT NULL,
          descripcion TEXT NOT NULL,
          monto REAL NOT NULL,
          fecha TEXT NOT NULL,
          notas TEXT,
          FOREIGN KEY (cultivo_id) REFERENCES cultivos (id) ON DELETE CASCADE
        )
      ''');
      print('Tabla egresos creada exitosamente');
    } else {
      // Verificar si la tabla tiene las columnas correctas
      final columns = await db.rawQuery('PRAGMA table_info(egresos)');
      final columnNames = columns.map((c) => c['name'] as String).toList();

      // Si no tiene las columnas correctas, eliminar y recrear
      if (!columnNames.contains('cultivo_id') ||
          !columnNames.contains('cultivo_nombre')) {
        print('Tabla egresos con estructura incorrecta, recreando...');
        await db.execute('DROP TABLE IF EXISTS egresos');

        await db.execute('''
          CREATE TABLE IF NOT EXISTS egresos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cultivo_id INTEGER NOT NULL,
            cultivo_nombre TEXT NOT NULL,
            tipo TEXT NOT NULL,
            descripcion TEXT NOT NULL,
            monto REAL NOT NULL,
            fecha TEXT NOT NULL,
            notas TEXT,
            FOREIGN KEY (cultivo_id) REFERENCES cultivos (id) ON DELETE CASCADE
          )
        ''');
        print('Tabla egresos recreada con estructura correcta');
      }
    }
  }

  // ========== 🆕 MÉTODOS DE ALERTAS ==========

  Future<int> insertarAlerta(Map<String, Object?> alerta) async {
    final db = await openDataBase();
    return await db.insert('alertas', alerta);
  }

  Future<List<Map<String, Object?>>> getAllAlertas() async {
    final db = await openDataBase();
    return await db.query(
      'alertas',
      orderBy: 'fecha DESC',
      where: 'resuelta = 0',
    );
  }

  Future<List<Map<String, Object?>>> getAlertasPorSeveridad(
    String severidad,
  ) async {
    final db = await openDataBase();
    return await db.query(
      'alertas',
      where: 'severidad = ? AND resuelta = 0',
      whereArgs: [severidad],
      orderBy: 'fecha DESC',
    );
  }

  Future<int> marcarAlertaResuelta(int id) async {
    final db = await openDataBase();
    return await db.update(
      'alertas',
      {'resuelta': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> eliminarAlerta(int id) async {
    final db = await openDataBase();
    return await db.delete('alertas', where: 'id = ?', whereArgs: [id]);
  }

  // ========== 🆕 MÉTODOS DE TAREAS ==========

  Future<int> insertarTarea(Map<String, Object?> tarea) async {
    final db = await openDataBase();
    return await db.insert('tareas', tarea);
  }

  Future<List<Map<String, Object?>>> getAllTareas() async {
    final db = await openDataBase();
    return await db.query('tareas', orderBy: 'fechaProgramada ASC');
  }

  Future<List<Map<String, Object?>>> getTareasPendientes() async {
    final db = await openDataBase();
    return await db.query(
      'tareas',
      where: 'completada = 0',
      orderBy: 'fechaProgramada ASC',
    );
  }

  Future<List<Map<String, Object?>>> getTareasHoy() async {
    final db = await openDataBase();
    final hoy = DateTime.now().toIso8601String().split('T')[0];
    return await db.query(
      'tareas',
      where: 'completada = 0 AND fechaProgramada LIKE ?',
      whereArgs: ['$hoy%'],
    );
  }

  Future<int> marcarTareaCompletada(int id) async {
    final db = await openDataBase();
    return await db.update(
      'tareas',
      {'completada': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> eliminarTarea(int id) async {
    final db = await openDataBase();
    return await db.delete('tareas', where: 'id = ?', whereArgs: [id]);
  }

  // ========== 🆕 GENERACIÓN AUTOMÁTICA DE ALERTAS ==========

  Future<void> generarAlertasAutomaticas() async {
    final cultivos = await getAllCultivos();
    final ahora = DateTime.now();

    for (final cultivoMap in cultivos) {
      final cultivoId = cultivoMap['id'] as int;
      final nombre = cultivoMap['nombre'] as String;
      final estado = (cultivoMap['estado'] as String).toLowerCase();
      final fechaCosecha = cultivoMap['fechaCosecha'] as String?;
      final isRisk = (cultivoMap['isRisk'] as int?) == 1;

      // 1. Alerta de riesgo
      if (isRisk && estado == 'en_riesgo') {
        final razonRiesgo =
            cultivoMap['riskReason'] as String? ?? 'Sin especificar';
        await insertarAlerta({
          'cultivoId': cultivoId,
          'tipo': 'riesgo',
          'severidad': 'critical',
          'titulo': 'Cultivo en riesgo',
          'mensaje': 'El cultivo "$nombre" presenta: $razonRiesgo',
          'fecha': ahora.toIso8601String(),
          'rutaDestino': '/cultivos',
        });
      }

      // 2. Alerta de cosecha próxima
      if (fechaCosecha != null && estado == 'activo') {
        final fechaCosechaDate = DateTime.tryParse(fechaCosecha);
        if (fechaCosechaDate != null) {
          final diasRestantes = fechaCosechaDate.difference(ahora).inDays;
          if (diasRestantes >= 0 && diasRestantes <= 7) {
            await insertarAlerta({
              'cultivoId': cultivoId,
              'tipo': 'cosecha',
              'severidad': diasRestantes <= 2 ? 'warning' : 'info',
              'titulo': 'Cosecha próxima',
              'mensaje':
                  'El cultivo "$nombre" estará listo para cosechar en $diasRestantes días',
              'fecha': ahora.toIso8601String(),
              'rutaDestino': '/cosecha',
            });
          }
        }
      }
    }
  }

  // ========== MÉTODOS DE CONVERSACIONES ==========

  Future<int> crearConversacion(String titulo) async {
    final db = await openDataBase();
    final now = DateTime.now().toIso8601String();
    return await db.insert('conversaciones', {
      'titulo': titulo,
      'fechaCreacion': now,
      'ultimaActualizacion': now,
      'mensajesCount': 0,
    });
  }

  Future<List<Map<String, Object?>>> getAllConversaciones() async {
    final db = await openDataBase();
    return await db.query(
      'conversaciones',
      orderBy: 'ultimaActualizacion DESC',
    );
  }

  Future<int> actualizarTituloConversacion(int id, String nuevoTitulo) async {
    final db = await openDataBase();
    return await db.update(
      'conversaciones',
      {
        'titulo': nuevoTitulo,
        'ultimaActualizacion': DateTime.now().toIso8601String(),
      },
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> eliminarConversacion(int id) async {
    final db = await openDataBase();
    return await db.delete('conversaciones', where: 'id = ?', whereArgs: [id]);
  }

  Future<int> insertarMensaje(Map<String, Object?> mensaje) async {
    final db = await openDataBase();
    final conversacionId = mensaje['conversacionId'] as int;
    final mensajeId = await db.insert('mensajes', mensaje);
    await db.rawUpdate(
      'UPDATE conversaciones SET mensajesCount = mensajesCount + 1, ultimaActualizacion = ? WHERE id = ?',
      [DateTime.now().toIso8601String(), conversacionId],
    );
    return mensajeId;
  }

  Future<List<Map<String, Object?>>> getMensajes(int conversacionId) async {
    final db = await openDataBase();
    return await db.query(
      'mensajes',
      where: 'conversacionId = ?',
      whereArgs: [conversacionId],
      orderBy: 'fecha ASC',
    );
  }

  Future<Map<String, Object?>?> getUltimoMensaje(int conversacionId) async {
    final db = await openDataBase();
    final result = await db.query(
      'mensajes',
      where: 'conversacionId = ?',
      whereArgs: [conversacionId],
      orderBy: 'fecha DESC',
      limit: 1,
    );
    return result.isNotEmpty ? result.first : null;
  }

  // ========== MÉTODOS DE CULTIVOS ==========

  Future<int> addData(Map<String, Object?> row) async {
    final db = await openDataBase();
    return await db.insert(
      'mitabla',
      row,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<int> insertCultivo(Map<String, Object?> cultivoRow) async {
    final db = await openDataBase();
    return await db.insert(
      'cultivos',
      cultivoRow,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Map<String, Object?>>> getAllCultivos() async {
    final db = await openDataBase();
    return await db.query('cultivos');
  }

  Future<int> updateCultivo(int id, Map<String, Object?> row) async {
    final db = await openDataBase();
    return await db.update('cultivos', row, where: 'id = ?', whereArgs: [id]);
  }

  Future<int> updateEstado(int id, String nuevoEstado) async {
    final db = await openDataBase();
    return await db.update(
      'cultivos',
      {'estado': nuevoEstado},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> deleteCultivo(int id) async {
    final db = await openDataBase();
    return await db.delete('cultivos', where: 'id = ?', whereArgs: [id]);
  }

  Future<int> insertTipoCultivo(Map<String, Object?> row) async {
    final db = await openDataBase();
    return await db.insert(
      'tipos_cultivo',
      row,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Map<String, Object?>>> getAllTiposCultivo() async {
    final db = await openDataBase();
    return await db.query('tipos_cultivo');
  }

  Future<int> deleteTipoCultivo(int id) async {
    final db = await openDataBase();
    return await db.delete('tipos_cultivo', where: 'id = ?', whereArgs: [id]);
  }

  Future<int> insertCategoria(Map<String, Object?> row) async {
    final db = await openDataBase();
    return await db.insert(
      'categorias',
      row,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Map<String, Object?>>> getAllCategorias() async {
    final db = await openDataBase();
    return await db.query('categorias');
  }

  Future<int> deleteCategoria(int id) async {
    final db = await openDataBase();
    return await db.delete('categorias', where: 'id = ?', whereArgs: [id]);
  }

  // ========== MÉTODOS DE VENTAS ==========

  Future<int> insertVenta(Map<String, Object?> ventaRow) async {
    final db = await openDataBase();
    return await db.insert(
      'ventas',
      ventaRow,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<Map<String, Object?>>> getAllVentas() async {
    final db = await openDataBase();
    return await db.query('ventas', orderBy: 'fecha DESC');
  }

  Future<int> updateVenta(int id, Map<String, Object?> row) async {
    final db = await openDataBase();
    return await db.update('ventas', row, where: 'id = ?', whereArgs: [id]);
  }

  Future<int> deleteVenta(int id) async {
    final db = await openDataBase();
    return await db.delete('ventas', where: 'id = ?', whereArgs: [id]);
  }

  Future<double> getTotalVentas() async {
    final db = await openDataBase();
    final result = await db.rawQuery('SELECT SUM(total) as total FROM ventas');
    return (result.first['total'] as num?)?.toDouble() ?? 0.0;
  }

  Future<List<Map<String, Object?>>> getVentasPorCultivo(int cultivoId) async {
    final db = await openDataBase();
    return await db.query(
      'ventas',
      where: 'cultivoId = ?',
      whereArgs: [cultivoId],
      orderBy: 'fecha DESC',
    );
  }

  Future<List<Map<String, Object?>>> getVentasPorFechas(
    String fechaInicio,
    String fechaFin,
  ) async {
    final db = await openDataBase();
    return await db.query(
      'ventas',
      where: 'fecha BETWEEN ? AND ?',
      whereArgs: [fechaInicio, fechaFin],
      orderBy: 'fecha DESC',
    );
  }

  // ========== MÉTODOS DE USUARIOS ==========

  Future<Map<String, dynamic>> registrarUsuario(
    String nombre,
    String correo,
    String password,
  ) async {
    final db = await openDataBase();
    final existingUser = await db.query(
      'usuarios',
      where: 'correo = ?',
      whereArgs: [correo],
    );
    if (existingUser.isNotEmpty)
      throw Exception('Ya existe un usuario con este correo');

    final passwordHash = _hashPassword(password);
    final id = await db.insert('usuarios', {
      'nombre': nombre,
      'correo': correo,
      'passwordHash': passwordHash,
    });
    return {'id': id, 'nombre': nombre, 'correo': correo};
  }

  Future<Map<String, dynamic>> iniciarSesion(
    String correo,
    String password,
  ) async {
    final db = await openDataBase();
    final result = await db.query(
      'usuarios',
      where: 'correo = ?',
      whereArgs: [correo],
    );
    if (result.isEmpty) throw Exception('Usuario o contraseña incorrectos');

    final user = result.first;
    final storedHash = user['passwordHash'] as String;
    if (!_verifyPassword(password, storedHash))
      throw Exception('Usuario o contraseña incorrectos');

    return {
      'id': user['id'],
      'nombre': user['nombre'],
      'correo': user['correo'],
    };
  }

  Future<void> generarTokenRecuperacion(String correo) async {
    final db = await openDataBase();
    final result = await db.query(
      'usuarios',
      where: 'correo = ?',
      whereArgs: [correo],
    );
    if (result.isEmpty) return;

    final token = (100000 + DateTime.now().millisecondsSinceEpoch % 900000)
        .toString()
        .substring(0, 6);
    final expiryTime = DateTime.now()
        .add(const Duration(hours: 1))
        .millisecondsSinceEpoch;
    await db.update(
      'usuarios',
      {'resetToken': token, 'resetTokenExpiry': expiryTime},
      where: 'correo = ?',
      whereArgs: [correo],
    );
    print('Token de recuperación para $correo: $token');
  }

  Future<bool> verificarTokenRecuperacion(String correo, String token) async {
    final db = await openDataBase();
    final result = await db.query(
      'usuarios',
      columns: ['resetToken', 'resetTokenExpiry'],
      where: 'correo = ?',
      whereArgs: [correo],
    );
    if (result.isEmpty) return false;

    final storedToken = result.first['resetToken'] as String?;
    final expiryTime = result.first['resetTokenExpiry'] as int?;
    if (storedToken == null || expiryTime == null) return false;

    return storedToken == token &&
        DateTime.now().millisecondsSinceEpoch < expiryTime;
  }

  Future<void> actualizarContrasena(String correo, String nuevaPassword) async {
    final db = await openDataBase();
    final newHash = _hashPassword(nuevaPassword);
    await db.update(
      'usuarios',
      {'passwordHash': newHash, 'resetToken': null, 'resetTokenExpiry': null},
      where: 'correo = ?',
      whereArgs: [correo],
    );
  }

  String _hashPassword(String password) {
    final bytes = utf8.encode(password);
    final digest = sha256.convert(bytes);
    return digest.toString();
  }

  bool _verifyPassword(String password, String storedHash) {
    return _hashPassword(password) == storedHash;
  }

  Future<List<Map<String, Object?>>> getAllData() async {
    final db = await openDataBase();
    return await db.query('mitabla');
  }

  Future<int> deleteById(int id) async {
    final db = await openDataBase();
    return await db.delete('mitabla', where: 'id = ?', whereArgs: [id]);
  }

  Future<void> close() async {
    if (_database != null) {
      await _database!.close();
      _database = null;
    }
  }

  // ========== MÉTODOS DE CRONOGRAMA DE ACTIVIDADES ==========

  Future<int> insertarCronogramaActividad(
    Map<String, Object?> actividad,
  ) async {
    final db = await openDataBase();
    return await db.insert('cronograma_actividades', actividad);
  }

  Future<List<Map<String, Object?>>> getAllCronogramaActividades() async {
    final db = await openDataBase();
    return await db.query(
      'cronograma_actividades',
      orderBy: 'fechaProgramada ASC',
    );
  }

  Future<List<Map<String, Object?>>> getCronogramaActividadesPorCultivo(
    int cultivoId,
  ) async {
    final db = await openDataBase();
    return await db.query(
      'cronograma_actividades',
      where: 'cultivoId = ?',
      whereArgs: [cultivoId],
      orderBy: 'fechaProgramada ASC',
    );
  }

  Future<List<Map<String, Object?>>> getCronogramaActividadesPorEstado(
    String estado,
  ) async {
    final db = await openDataBase();
    return await db.query(
      'cronograma_actividades',
      where: 'estado = ?',
      whereArgs: [estado],
      orderBy: 'fechaProgramada ASC',
    );
  }

  Future<List<Map<String, Object?>>>
  getCronogramaActividadesPendientes() async {
    final db = await openDataBase();
    return await db.query(
      'cronograma_actividades',
      where: 'estado = ? OR estado = ?',
      whereArgs: ['pendiente', 'en_progreso'],
      orderBy: 'fechaProgramada ASC',
    );
  }

  Future<List<Map<String, Object?>>> getCronogramaActividadesHoy() async {
    final db = await openDataBase();
    final hoy = DateTime.now().toIso8601String().split('T')[0];
    return await db.query(
      'cronograma_actividades',
      where: 'fechaProgramada LIKE ? AND (estado = ? OR estado = ?)',
      whereArgs: ['$hoy%', 'pendiente', 'en_progreso'],
      orderBy: 'fechaProgramada ASC',
    );
  }

  Future<int> actualizarCronogramaActividad(
    int id,
    Map<String, Object?> actividad,
  ) async {
    final db = await openDataBase();
    actividad['actualizadoEn'] = DateTime.now().toIso8601String();
    return await db.update(
      'cronograma_actividades',
      actividad,
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> marcarCronogramaActividadCompletada(int id) async {
    final db = await openDataBase();
    return await db.update(
      'cronograma_actividades',
      {
        'estado': 'completada',
        'fechaRealizada': DateTime.now().toIso8601String(),
        'actualizadoEn': DateTime.now().toIso8601String(),
      },
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> eliminarCronogramaActividad(int id) async {
    final db = await openDataBase();
    return await db.delete(
      'cronograma_actividades',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> getCronogramaActividadesCountPorCultivo(int cultivoId) async {
    final db = await openDataBase();
    final result = await db.rawQuery(
      'SELECT COUNT(*) as count FROM cronograma_actividades WHERE cultivoId = ?',
      [cultivoId],
    );
    return result.first['count'] as int;
  }

  Future<int> updateImagenPerfil(int userId, String imagenPath) async {
    final db = await openDataBase();
    return await db.update(
      'usuarios',
      {'imagenPerfil': imagenPath},
      where: 'id = ?',
      whereArgs: [userId],
    );
  }

  Future<Map<String, dynamic>?> getUsuario(int userId) async {
    final db = await openDataBase();
    final result = await db.query(
      'usuarios',
      where: 'id = ?',
      whereArgs: [userId],
      limit: 1,
    );
    return result.isNotEmpty ? result.first : null;
  }

  Future<double> getCronogramaCostosTotalesPorCultivo(int cultivoId) async {
    final db = await openDataBase();
    final result = await db.rawQuery(
      'SELECT SUM(costo) as total FROM cronograma_actividades WHERE cultivoId = ? AND costo IS NOT NULL',
      [cultivoId],
    );
    return (result.first['total'] as num?)?.toDouble() ?? 0.0;
  }

  // ========== 🆕 MÉTODOS DE EGRESOS ==========

  Future<int> insertEgreso(Map<String, Object?> egreso) async {
    final db = await openDataBase();

    // Asegurar que las claves coincidan con el esquema de la BD
    final egresoAjustado = {
      'cultivo_id': egreso['cultivoId'],
      'cultivo_nombre': egreso['cultivoNombre'],
      'tipo': egreso['tipo'],
      'descripcion': egreso['descripcion'],
      'monto': egreso['monto'],
      'fecha': egreso['fecha'],
      'notas': egreso['notas'],
    };

    return await db.insert('egresos', egresoAjustado);
  }

  Future<List<Map<String, Object?>>> getAllEgresos() async {
    final db = await openDataBase();
    return await db.query('egresos', orderBy: 'fecha DESC');
  }

  Future<List<Map<String, Object?>>> getEgresosByCultivo(int cultivoId) async {
    final db = await openDataBase();
    return await db.query(
      'egresos',
      where: 'cultivo_id = ?', // ✅ CORREGIDO
      whereArgs: [cultivoId],
      orderBy: 'fecha DESC',
    );
  }

  Future<int> updateEgreso(int id, Map<String, Object?> egreso) async {
    final db = await openDataBase();

    // Asegurar que las claves coincidan con el esquema de la BD
    final egresoAjustado = {
      'cultivo_id': egreso['cultivoId'],
      'cultivo_nombre': egreso['cultivoNombre'],
      'tipo': egreso['tipo'],
      'descripcion': egreso['descripcion'],
      'monto': egreso['monto'],
      'fecha': egreso['fecha'],
      'notas': egreso['notas'],
    };

    return await db.update(
      'egresos',
      egresoAjustado,
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> deleteEgreso(int id) async {
    final db = await openDataBase();
    return await db.delete('egresos', where: 'id = ?', whereArgs: [id]);
  }

  Future<double> getEgresosTotalesPorCultivo(int cultivoId) async {
    final db = await openDataBase();
    final result = await db.rawQuery(
      'SELECT SUM(monto) as total FROM egresos WHERE cultivo_id = ?', // ✅ CORREGIDO
      [cultivoId],
    );
    return (result.first['total'] as num?)?.toDouble() ?? 0.0;
  }

  Future<double> getEgresosTotalesGenerales() async {
    final db = await openDataBase();
    final result = await db.rawQuery('SELECT SUM(monto) as total FROM egresos');
    return (result.first['total'] as num?)?.toDouble() ?? 0.0;
  }

  Future<List<Map<String, Object?>>> getEgresosPorTipo(String tipo) async {
    final db = await openDataBase();
    return await db.query(
      'egresos',
      where: 'tipo = ?',
      whereArgs: [tipo],
      orderBy: 'fecha DESC',
    );
  }

  Future<List<Map<String, Object?>>> getEgresosPorRangoFechas(
    String fechaInicio,
    String fechaFin,
  ) async {
    final db = await openDataBase();
    return await db.rawQuery(
      'SELECT * FROM egresos WHERE fecha BETWEEN ? AND ? ORDER BY fecha DESC',
      [fechaInicio, fechaFin],
    );
  }
}
