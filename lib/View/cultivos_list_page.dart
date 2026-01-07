// lib/View/cultivos_list_page.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:amgeca/classes/tipo_cultivo.dart';
import 'package:amgeca/classes/categoria.dart';
import 'cultivo_form.dart';
import 'cronograma_page.dart';
import 'tipos_list_page.dart';
import 'categorias_list_page.dart';
import 'cosecha_form.dart';
import 'riesgo_form.dart';
import 'inactivacion_form.dart';
import 'egresos_page.dart';
import 'historial_riesgos_page.dart';

class CultivosListPage extends StatefulWidget {
  const CultivosListPage({Key? key}) : super(key: key);

  @override
  State<CultivosListPage> createState() => _CultivosListPageState();
}

class _CultivosListPageState extends State<CultivosListPage> {
  late Future<List<Cultivo>> _cultivosFuture;
  Map<int, String> _tipoMap = {};
  Map<int, String> _categoriaMap = {};
  String? _filtroActivo;
  int? _expandedCultivoId;

  @override
  void initState() {
    super.initState();
    _loadCultivos();
    _loadAuxData();
  }

  void _loadCultivos() {
    setState(() {
      _cultivosFuture = _getCultivosFromDB();
    });
  }

  Future<void> _loadAuxData() async {
    final tiposRows = await BasedatoHelper.instance.getAllTiposCultivo();
    final categoriasRows = await BasedatoHelper.instance.getAllCategorias();
    setState(() {
      _tipoMap = Map.fromEntries(
        tiposRows.map((r) {
          final t = TipoCultivo.fromMap(r);
          return MapEntry(t.id ?? 0, t.nombre);
        }),
      );
      _categoriaMap = Map.fromEntries(
        categoriasRows.map((r) {
          final c = Categoria.fromMap(r);
          return MapEntry(c.id ?? 0, c.nombre);
        }),
      );
    });
  }

  Future<List<Cultivo>> _getCultivosFromDB() async {
    final rows = await BasedatoHelper.instance.getAllCultivos();
    final cultivos = rows.map((r) => Cultivo.fromMap(r)).toList();
    return cultivos;
  }

  void _navigateToForm({Cultivo? cultivo}) async {
    final result = await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => CultivoFormPage(cultivo: cultivo),
      ),
    );
    if (result == true) {
      _loadCultivos();
    }
  }

  void _navigateToEgresos(Cultivo cultivo) async {
    final result = await Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => EgresosPage(cultivo: cultivo)),
    );
    if (result == true) {
      _loadCultivos();
    }
  }

  void _navigateToHistorialCompleto(List<Cultivo> cultivos) async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => HistorialRiesgosPage(cultivos: cultivos),
      ),
    );
  }

  Future<void> _deleteCultivo(int id) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 28),
            SizedBox(width: 12),
            Text('Eliminar cultivo'),
          ],
        ),
        content: const Text(
          '¿Estás seguro de que deseas eliminar este cultivo? Esta acción no se puede deshacer.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancelar'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
            ),
            child: const Text('Eliminar'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await BasedatoHelper.instance.deleteCultivo(id);
      _loadCultivos();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Row(
            children: [
              Icon(Icons.check_circle, color: Colors.white),
              SizedBox(width: 12),
              Text('Cultivo eliminado exitosamente'),
            ],
          ),
          backgroundColor: Colors.green,
        ),
      );
    }
  }

  Future<void> _cambiarEstado(Cultivo cultivo, String nuevoEstado) async {
    // No permitir reactivar cultivos cosechados
    if (cultivo.esCosechado && nuevoEstado == Cultivo.ESTADO_ACTIVO) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Row(
            children: [
              Icon(Icons.error_outline, color: Colors.white),
              SizedBox(width: 12),
              Text('No se puede reactivar un cultivo cosechado'),
            ],
          ),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    // Manejar cada transición de estado
    if (nuevoEstado == Cultivo.ESTADO_COSECHADO) {
      // Abrir formulario de cosecha
      final result = await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => CosechaFormPage(cultivo: cultivo),
        ),
      );
      if (result == true) {
        _loadCultivos();
      }
    } else if (nuevoEstado == Cultivo.ESTADO_EN_RIESGO) {
      // Abrir formulario de riesgo
      final result = await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => RiesgoFormPage(cultivo: cultivo),
        ),
      );
      if (result == true) {
        _loadCultivos();
      }
    } else if (nuevoEstado == Cultivo.ESTADO_INACTIVO) {
      // Abrir formulario de inactivación
      final result = await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => InactivacionFormPage(cultivo: cultivo),
        ),
      );
      if (result == true) {
        _loadCultivos();
      }
    } else {
      // Cambio directo de estado (ej: volver a activo)
      if (cultivo.estado == Cultivo.ESTADO_EN_RIESGO &&
          nuevoEstado == Cultivo.ESTADO_ACTIVO) {
        // Si viene de riesgo a activo, guardar historial del riesgo
        try {
          // Obtener el cultivo actualizado con datos de riesgo
          final cultivos = await BasedatoHelper.instance.getAllCultivos();
          final cultivoActual = cultivos.firstWhere(
            (c) => c['id'] == cultivo.id,
          );

          final cultivoObj = Cultivo.fromMap(cultivoActual);

          // Agregar riesgo actual al historial y limpiar campos de riesgo
          final datosActualizados = cultivoObj.agregarRiesgoAlHistorial();

          // Actualizar estado a activo
          datosActualizados['estado'] = nuevoEstado;

          await BasedatoHelper.instance.updateCultivo(
            cultivo.id!,
            datosActualizados,
          );

          _loadCultivos();
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(Icons.history, color: Colors.white),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Cultivo reactivado. Riesgo anterior guardado en historial.',
                      style: const TextStyle(fontSize: 14),
                    ),
                  ),
                ],
              ),
              backgroundColor: Colors.green,
              duration: const Duration(seconds: 4),
            ),
          );
        } catch (e) {
          // Si hay error, hacer cambio normal
          await BasedatoHelper.instance.updateEstado(cultivo.id!, nuevoEstado);
          _loadCultivos();
        }
      } else {
        // Cambio normal de estado
        await BasedatoHelper.instance.updateEstado(cultivo.id!, nuevoEstado);
        _loadCultivos();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Estado actualizado a: ${_getEstadoLabel(nuevoEstado)}',
            ),
            backgroundColor: Colors.green,
          ),
        );
      }
    }
  }

  String _getEstadoLabel(String estado) {
    switch (estado) {
      case Cultivo.ESTADO_ACTIVO:
        return 'Activo';
      case Cultivo.ESTADO_EN_RIESGO:
        return 'En Riesgo';
      case Cultivo.ESTADO_COSECHADO:
        return 'Cosechado';
      case Cultivo.ESTADO_INACTIVO:
        return 'Inactivo';
      default:
        return estado;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mis Cultivos'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        elevation: 0,
        leading: const SizedBox(),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadCultivos,
            tooltip: 'Actualizar',
          ),
        ],
      ),
      body: FutureBuilder<List<Cultivo>>(
        future: _cultivosFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: CircularProgressIndicator(color: Colors.green),
            );
          }
          if (snapshot.hasError) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 64, color: Colors.red),
                  const SizedBox(height: 16),
                  Text('Error: ${snapshot.error}'),
                ],
              ),
            );
          }
          final cultivos = snapshot.data ?? [];
          if (cultivos.isEmpty) {
            return _buildEmptyState();
          }

          return _buildCultivosList(cultivos);
        },
      ),
      floatingActionButton: _filtroActivo == null
          ? FloatingActionButton.extended(
              onPressed: () => _navigateToForm(),
              icon: const Icon(Icons.add),
              label: const Text('Nuevo'),
              backgroundColor: const Color.fromARGB(141, 3, 126, 8),
              foregroundColor: Colors.white,
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
    );
  }

  Widget _buildFilteredEmptyState() {
    final estadoEtiqueta = _filtroActivo != null
        ? _getEstadoLabel(_filtroActivo!)
        : 'esta sección';
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.info_outline, size: 80, color: Colors.grey),
            const SizedBox(height: 16),
            Text(
              'Aún no hay cultivos agregados a esta sección',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.grey[800],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Actualmente estás viendo: $estadoEtiqueta',
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            OutlinedButton.icon(
              onPressed: () {
                setState(() {
                  _filtroActivo = null;
                  _loadCultivos();
                });
              },
              icon: const Icon(Icons.clear_all),
              label: const Text('Mostrar todos los cultivos'),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.green,
                side: const BorderSide(color: Colors.green),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                color: Colors.green[50],
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.agriculture,
                size: 80,
                color: Colors.green[700],
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'No hay cultivos registrados',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.grey[800],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Comienza agregando tu primer cultivo',
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: () => _navigateToForm(),
              icon: const Icon(Icons.add),
              label: const Text('Crear Primer Cultivo'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 32,
                  vertical: 16,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _navigateToHistorialRiesgos(Cultivo cultivo) async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => HistorialRiesgosPage(cultivos: [cultivo]),
      ),
    );
  }

  Widget _buildCultivosList(List<Cultivo> cultivos) {
    final activos = cultivos.where((c) => c.esActivo).length;
    final cosechados = cultivos.where((c) => c.esCosechado).length;
    final enRiesgo = cultivos.where((c) => c.esEnRiesgo).length;
    final inactivos = cultivos.where((c) => c.esInactivo).length;

    // Contar cultivos únicos con riesgos (actuales o históricos)
    final cultivosConRiesgos = cultivos
        .where((c) => c.esEnRiesgo || c.tieneHistorialRiesgos)
        .length;

    final List<Cultivo> displayCultivos = _filtroActivo == null
        ? _ordenarCultivosPorEstado(cultivos)
        : cultivos.where((c) => c.estado == _filtroActivo).toList();

    return Column(
      children: [
        // Resumen con filtros
        Container(
          decoration: BoxDecoration(
            color: Colors.green,
            boxShadow: [
              BoxShadow(
                color: Colors.green.withOpacity(0.3),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Resumen de Cultivos',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 16),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      _buildSummaryCard(
                        '$activos',
                        'Activos',
                        Icons.check_circle,
                        Colors.green[100]!,
                        Colors.green[900]!,
                        onTap: () {
                          setState(() {
                            _filtroActivo =
                                _filtroActivo == Cultivo.ESTADO_ACTIVO
                                ? null
                                : Cultivo.ESTADO_ACTIVO;
                            _loadCultivos();
                          });
                        },
                        isActive: _filtroActivo == Cultivo.ESTADO_ACTIVO,
                      ),
                      _buildSummaryCard(
                        '$enRiesgo',
                        'En Riesgo',
                        Icons.warning,
                        Colors.orange[100]!,
                        Colors.orange[900]!,
                        onTap: () {
                          setState(() {
                            _filtroActivo =
                                _filtroActivo == Cultivo.ESTADO_EN_RIESGO
                                ? null
                                : Cultivo.ESTADO_EN_RIESGO;
                            _loadCultivos();
                          });
                        },
                        isActive: _filtroActivo == Cultivo.ESTADO_EN_RIESGO,
                      ),
                      _buildSummaryCard(
                        '$cosechados',
                        'Cosechados',
                        Icons.agriculture,
                        Colors.amber[100]!,
                        Colors.amber[900]!,
                        onTap: () {
                          setState(() {
                            _filtroActivo =
                                _filtroActivo == Cultivo.ESTADO_COSECHADO
                                ? null
                                : Cultivo.ESTADO_COSECHADO;
                            _loadCultivos();
                          });
                        },
                        isActive: _filtroActivo == Cultivo.ESTADO_COSECHADO,
                      ),
                      _buildSummaryCard(
                        '$inactivos',
                        'Inactivos',
                        Icons.cancel,
                        Colors.grey[300]!,
                        Colors.grey[900]!,
                        onTap: () {
                          setState(() {
                            _filtroActivo =
                                _filtroActivo == Cultivo.ESTADO_INACTIVO
                                ? null
                                : Cultivo.ESTADO_INACTIVO;
                            _loadCultivos();
                          });
                        },
                        isActive: _filtroActivo == Cultivo.ESTADO_INACTIVO,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),

        // 🆕 NUEVA SECCIÓN: Botón de configuración (solo cuando no hay filtro)
        if (_filtroActivo == null) ...[
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                // Botón gestionar tipos
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const TiposListPage(),
                        ),
                      );
                    },
                    icon: const Icon(Icons.category, size: 18),
                    label: const Text('Gestionar Tipos'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.blue,
                      side: const BorderSide(color: Colors.blue, width: 2),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ),

                const SizedBox(width: 12),

                // Botón gestionar categorías
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const CategoriasListPage(),
                        ),
                      );
                    },
                    icon: const Icon(Icons.label, size: 18),
                    label: const Text('Gest. Categorías'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.purple,
                      side: const BorderSide(color: Colors.purple, width: 2),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],

        // 🆕 NUEVA SECCIÓN: Botón de historial completo y filtro (solo en En Riesgo y Activos)
        if (_filtroActivo != null) ...[
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                // Botón limpiar filtro
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      setState(() {
                        _filtroActivo = null;
                        _loadCultivos();
                      });
                    },
                    icon: const Icon(Icons.clear_all, size: 18),
                    label: Text('Mostrar todos (${cultivos.length})'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.green,
                      side: const BorderSide(color: Colors.green, width: 2),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ),

                const SizedBox(width: 12),

                // 🆕 BOTÓN DE HISTORIAL COMPLETO (solo en En Riesgo)
                if (_filtroActivo == Cultivo.ESTADO_EN_RIESGO)
                  FloatingActionButton.extended(
                    onPressed: cultivosConRiesgos > 0
                        ? () => _navigateToHistorialCompleto(cultivos)
                        : null,
                    backgroundColor: cultivosConRiesgos > 0
                        ? Colors.orange[600]
                        : Colors.grey[300],
                    foregroundColor: Colors.white,
                    icon: Stack(
                      clipBehavior: Clip.none,
                      children: [
                        const Icon(Icons.history),
                        if (cultivosConRiesgos > 0)
                          Positioned(
                            right: -8,
                            top: -8,
                            child: Container(
                              padding: const EdgeInsets.all(4),
                              decoration: BoxDecoration(
                                color: Colors.red,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                '$cultivosConRiesgos',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                    label: Text(
                      cultivosConRiesgos > 0 ? 'Historial' : 'Sin riesgos',
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],

        // Lista de cultivos
        Expanded(
          child: displayCultivos.isEmpty
              ? _buildFilteredEmptyState()
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  itemCount: displayCultivos.length,
                  itemBuilder: (context, index) {
                    final cultivo = displayCultivos[index];
                    return _buildCultivoCard(cultivo);
                  },
                ),
        ),
      ],
    );
  }

  Widget _buildSummaryCard(
    String number,
    String label,
    IconData icon,
    Color bgColor,
    Color textColor, {
    VoidCallback? onTap,
    bool isActive = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(right: 12),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(16),
          child: Container(
            width: 120,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(16),
              border: isActive ? Border.all(color: textColor, width: 3) : null,
              boxShadow: isActive
                  ? [
                      BoxShadow(
                        color: textColor.withOpacity(0.3),
                        blurRadius: 8,
                        offset: const Offset(0, 4),
                      ),
                    ]
                  : null,
            ),
            child: Column(
              children: [
                Icon(icon, color: textColor, size: 28),
                const SizedBox(height: 8),
                Text(
                  number,
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  label,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: textColor,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCultivoCard(Cultivo cultivo) {
    final tipoNombre = _tipoMap[cultivo.tipoId ?? 0] ?? 'Sin tipo';
    final categoriaNombre =
        _categoriaMap[cultivo.categoriaId ?? 0] ?? 'Sin categoría';

    Color colorEstado;
    IconData iconoEstado;
    String labelEstado;

    switch (cultivo.estado) {
      case Cultivo.ESTADO_ACTIVO:
        colorEstado = Colors.green;
        iconoEstado = Icons.check_circle;
        labelEstado = 'ACTIVO';
        break;
      case Cultivo.ESTADO_EN_RIESGO:
        colorEstado = Colors.orange;
        iconoEstado = Icons.warning;
        labelEstado = 'EN RIESGO';
        break;
      case Cultivo.ESTADO_COSECHADO:
        colorEstado = Colors.amber;
        iconoEstado = Icons.agriculture;
        labelEstado = 'COSECHADO';
        break;
      case Cultivo.ESTADO_INACTIVO:
        colorEstado = Colors.grey;
        iconoEstado = Icons.cancel;
        labelEstado = 'INACTIVO';
        break;
      default:
        colorEstado = Colors.grey;
        iconoEstado = Icons.help;
        labelEstado = cultivo.estado.toUpperCase();
    }

    final bool showStateActions = !cultivo.esCosechado && !cultivo.esInactivo;
    final bool isExpanded = _expandedCultivoId == cultivo.id;

    return GestureDetector(
      onTap: () {
        setState(() {
          _expandedCultivoId = isExpanded ? null : cultivo.id;
        });
      },
      child: Card(
        margin: const EdgeInsets.only(bottom: 16),
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: colorEstado.withOpacity(0.5), width: 2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Imagen del cultivo
            if (cultivo.imagenUrl != null &&
                File(cultivo.imagenUrl!).existsSync())
              GestureDetector(
                onTap: () => _mostrarImagenCompleta(cultivo.imagenUrl!),
                child: ClipRRect(
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(16),
                  ),
                  child: Stack(
                    children: [
                      Image.file(
                        File(cultivo.imagenUrl!),
                        width: double.infinity,
                        height: 200,
                        fit: BoxFit.cover,
                      ),
                      Positioned(
                        top: 12,
                        right: 12,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            // Badge de estado
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: colorEstado,
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    iconoEstado,
                                    color: Colors.white,
                                    size: 16,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    labelEstado,
                                    style: const TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            // Botón de historial (solo para cultivos en riesgo con historial)
                            if (cultivo.esEnRiesgo &&
                                cultivo.tieneHistorialRiesgos) ...[
                              const SizedBox(height: 8),
                              Container(
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.9),
                                  borderRadius: BorderRadius.circular(20),
                                  border: Border.all(
                                    color: Colors.orange.withOpacity(0.5),
                                  ),
                                ),
                                child: IconButton(
                                  icon: const Icon(
                                    Icons.history,
                                    color: Colors.orange,
                                    size: 18,
                                  ),
                                  onPressed: () =>
                                      _navigateToHistorialRiesgos(cultivo),
                                  tooltip: 'Historial de Riesgos',
                                  padding: const EdgeInsets.all(8),
                                  constraints: const BoxConstraints(
                                    minWidth: 36,
                                    minHeight: 36,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            // Detailed summary section (shown when expanded)
            if (isExpanded) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey[50],
                  borderRadius: const BorderRadius.only(
                    bottomLeft: Radius.circular(16),
                    bottomRight: Radius.circular(16),
                  ),
                  border: Border(
                    top: BorderSide(color: colorEstado.withOpacity(0.3)),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Summary header
                    Row(
                      children: [
                        Icon(Icons.info_outline, color: colorEstado, size: 20),
                        const SizedBox(width: 8),
                        Text(
                          'Resumen del Cultivo',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: colorEstado,
                          ),
                        ),
                        const Spacer(),
                        // Cronograma button (only for active and at-risk crops)
                        if (cultivo.esActivo || cultivo.esEnRiesgo)
                          ElevatedButton.icon(
                            onPressed: () => _mostrarCronograma(cultivo),
                            icon: const Icon(Icons.schedule, size: 16),
                            label: const Text('Cronograma'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: colorEstado,
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
                              textStyle: const TextStyle(fontSize: 12),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // Detailed information grid
                    _buildDetailedInfoGrid(cultivo),

                    const SizedBox(height: 16),

                    // Progress indicators
                    _buildProgressIndicators(cultivo),
                  ],
                ),
              ),
            ],

            // Contenido de la tarjeta
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Encabezado
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: colorEstado.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Icon(
                          Icons.agriculture,
                          color: colorEstado,
                          size: 24,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              cultivo.nombre,
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              '$tipoNombre • $categoriaNombre',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                      ),
                      if (cultivo.imagenUrl == null)
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: colorEstado,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(iconoEstado, color: Colors.white, size: 14),
                              const SizedBox(width: 4),
                              Text(
                                labelEstado,
                                style: const TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 16),

                  // Información adicional
                  _buildInfoRow(
                    Icons.straighten,
                    'Área',
                    '${cultivo.area.toStringAsFixed(0)} ha',
                  ),
                  const SizedBox(height: 8),
                  _buildInfoRow(Icons.water, 'Riego', cultivo.tipoRiego),
                  const SizedBox(height: 8),
                  _buildInfoRow(Icons.landscape, 'Suelo', cultivo.tipoSuelo),
                  const SizedBox(height: 16),

                  // Fechas
                  Row(
                    children: [
                      Expanded(
                        child: _buildDateChip(
                          'Siembra',
                          cultivo.fechaSiembra,
                          Colors.blue,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: _buildDateChip(
                          'Cosecha Estimada',
                          cultivo.fechaCosecha ?? 'Por definir',
                          Colors.green,
                        ),
                      ),
                    ],
                  ),

                  // Notas
                  if (cultivo.notas != null && cultivo.notas!.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.grey[100],
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: [
                          Icon(Icons.notes, size: 16, color: Colors.grey[600]),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              cultivo.notas!,
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey[700],
                                fontStyle: FontStyle.italic,
                              ),
                              maxLines: 3,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],

                  // Información de riesgo
                  if (cultivo.esEnRiesgo &&
                      cultivo.razonRiesgo != null &&
                      cultivo.razonRiesgo!.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.orange[50],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.orange),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Icon(
                                Icons.warning,
                                size: 18,
                                color: Colors.orange,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Riesgo: ${cultivo.tipoRiesgo ?? "No especificado"}',
                                style: const TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.orange,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Text(
                            cultivo.razonRiesgo!,
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.orange[900],
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                          if (cultivo.fechaRiesgo != null) ...[
                            const SizedBox(height: 4),
                            Text(
                              'Registrado: ${cultivo.fechaRiesgo}',
                              style: TextStyle(
                                fontSize: 10,
                                color: Colors.orange[700],
                                fontStyle: FontStyle.italic,
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ],

                  const SizedBox(height: 16),
                  const Divider(),
                  const SizedBox(height: 8),

                  // Botones de acción
                  Row(
                    children: [
                      // Botón Cosechar/Activar (según estado)
                      if (showStateActions)
                        Expanded(
                          child: _buildActionButton(
                            label: cultivo.esActivo ? 'Cosechar' : 'Activar',
                            icon: cultivo.esActivo
                                ? Icons.agriculture
                                : Icons.play_arrow,
                            color: Colors.green,
                            onPressed: () async {
                              if (cultivo.esActivo) {
                                _cambiarEstado(
                                  cultivo,
                                  Cultivo.ESTADO_COSECHADO,
                                );
                              } else {
                                // Mostrar confirmación antes de activar
                                final confirmed = await showDialog<bool>(
                                  context: context,
                                  builder: (context) => AlertDialog(
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(16),
                                    ),
                                    title: Row(
                                      children: [
                                        Container(
                                          padding: const EdgeInsets.all(8),
                                          decoration: BoxDecoration(
                                            color: Colors.green[100],
                                            shape: BoxShape.circle,
                                          ),
                                          child: Icon(
                                            Icons.play_arrow,
                                            color: Colors.green[700],
                                            size: 24,
                                          ),
                                        ),
                                        const SizedBox(width: 12),
                                        const Text('Activar Cultivo'),
                                      ],
                                    ),
                                    content: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          '¿Estás seguro de que deseas activar el cultivo "${cultivo.nombre}"?',
                                          style: const TextStyle(fontSize: 16),
                                        ),
                                        const SizedBox(height: 12),
                                        Container(
                                          padding: const EdgeInsets.all(12),
                                          decoration: BoxDecoration(
                                            color: Colors.green[50],
                                            borderRadius: BorderRadius.circular(
                                              8,
                                            ),
                                            border: Border.all(
                                              color: Colors.green[200]!,
                                            ),
                                          ),
                                          child: Row(
                                            children: [
                                              Icon(
                                                Icons.info_outline,
                                                color: Colors.green[700],
                                                size: 20,
                                              ),
                                              const SizedBox(width: 8),
                                              Expanded(
                                                child: Text(
                                                  'El cultivo pasará a estado activo y estará disponible para operaciones.',
                                                  style: TextStyle(
                                                    fontSize: 13,
                                                    color: Colors.green[700],
                                                  ),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ],
                                    ),
                                    actions: [
                                      TextButton(
                                        onPressed: () =>
                                            Navigator.of(context).pop(false),
                                        style: TextButton.styleFrom(
                                          foregroundColor: Colors.grey[600],
                                        ),
                                        child: const Text('Cancelar'),
                                      ),
                                      ElevatedButton(
                                        onPressed: () =>
                                            Navigator.of(context).pop(true),
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: Colors.green,
                                          foregroundColor: Colors.white,
                                          shape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(
                                              8,
                                            ),
                                          ),
                                        ),
                                        child: const Text('Activar'),
                                      ),
                                    ],
                                  ),
                                );

                                if (confirmed == true) {
                                  _cambiarEstado(
                                    cultivo,
                                    Cultivo.ESTADO_ACTIVO,
                                  );
                                }
                              }
                            },
                          ),
                        ),

                      // Mostrar mensaje si ya está cosechado
                      if (cultivo.esCosechado)
                        Expanded(
                          child: Container(
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            decoration: BoxDecoration(
                              color: Colors.amber[50],
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: Colors.amber),
                            ),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.check_circle,
                                  color: Colors.amber,
                                  size: 16,
                                ),
                                SizedBox(width: 6),
                                Text(
                                  'Cosechado',
                                  style: TextStyle(
                                    color: Colors.amber,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 13,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                      if (showStateActions) const SizedBox(width: 8),

                      // Botón Riesgo/Inactivar (dependiendo del estado)
                      if (showStateActions)
                        Expanded(
                          child: _buildActionButton(
                            label: cultivo.esEnRiesgo ? 'Inactivar' : 'Riesgo',
                            icon: cultivo.esEnRiesgo
                                ? Icons.cancel
                                : Icons.warning,
                            color: cultivo.esEnRiesgo
                                ? Colors.grey
                                : Colors.orange,
                            onPressed: () {
                              if (cultivo.esEnRiesgo) {
                                // Si está en riesgo, permitir inactivar
                                _cambiarEstado(
                                  cultivo,
                                  Cultivo.ESTADO_INACTIVO,
                                );
                              } else {
                                // Si no está en riesgo, marcar como riesgo
                                _cambiarEstado(
                                  cultivo,
                                  Cultivo.ESTADO_EN_RIESGO,
                                );
                              }
                            },
                          ),
                        ),
                      if (showStateActions) const SizedBox(width: 8),
                      if (!cultivo.esInactivo)
                        IconButton(
                          icon: const Icon(Icons.edit, color: Colors.blue),
                          onPressed: () => _navigateToForm(cultivo: cultivo),
                          tooltip: 'Editar',
                        ),
                      // Botón de egresos solo para cultivos activos y en riesgo
                      if (cultivo.esActivo || cultivo.esEnRiesgo)
                        IconButton(
                          icon: const Icon(
                            Icons.money_off,
                            color: Colors.purple,
                          ),
                          onPressed: () => _navigateToEgresos(cultivo),
                          tooltip: 'Egresos',
                        ),
                      IconButton(
                        icon: const Icon(Icons.delete, color: Colors.red),
                        onPressed: () => _deleteCultivo(cultivo.id!),
                        tooltip: 'Eliminar',
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 16, color: Colors.grey[600]),
        const SizedBox(width: 8),
        Text(
          '$label:',
          style: TextStyle(
            fontSize: 13,
            color: Colors.grey[600],
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          value,
          style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _buildDateChip(String label, String date, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            date,
            style: TextStyle(
              fontSize: 12,
              color: color,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton({
    required String label,
    required IconData icon,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return OutlinedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 16),
      label: Text(label),
      style: OutlinedButton.styleFrom(
        foregroundColor: color,
        side: BorderSide(color: color),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(vertical: 8),
      ),
    );
  }

  void _mostrarImagenCompleta(String imagePath) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.file(File(imagePath), fit: BoxFit.contain),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.close),
              label: const Text('Cerrar'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: Colors.black,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Helper methods for the detailed summary
  Widget _buildDetailedInfoGrid(Cultivo cultivo) {
    final tipoNombre = _tipoMap[cultivo.tipoId ?? 0] ?? 'Sin tipo';
    final categoriaNombre =
        _categoriaMap[cultivo.categoriaId ?? 0] ?? 'Sin categoría';

    return Column(
      children: [
        // Two-column info grid
        Row(
          children: [
            Expanded(
              child: _buildDetailItem(Icons.category, 'Tipo', tipoNombre),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildDetailItem(
                Icons.label,
                'Categoría',
                categoriaNombre,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _buildDetailItem(
                Icons.straighten,
                'Área Total',
                '${cultivo.area.toStringAsFixed(1)} ha',
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildDetailItem(
                Icons.water_drop,
                'Tipo de Riego',
                cultivo.tipoRiego,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _buildDetailItem(
                Icons.landscape,
                'Tipo de Suelo',
                cultivo.tipoSuelo,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildDetailItem(
                Icons.calendar_today,
                'Estado',
                cultivo.estado.toUpperCase(),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildDetailItem(IconData icon, String label, String value) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey[300]!),
      ),
      child: Row(
        children: [
          Icon(icon, size: 16, color: Colors.grey[600]),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey[600],
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressIndicators(Cultivo cultivo) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Progreso del Cultivo',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: Colors.grey[700],
          ),
        ),
        const SizedBox(height: 8),

        // Progress timeline
        _buildProgressItem(
          'Siembra',
          cultivo.fechaSiembra,
          Icons.agriculture,
          Colors.green,
          isCompleted: true,
        ),

        if (cultivo.fechaCosecha != null) ...[
          _buildProgressItem(
            'Cosecha Estimada',
            cultivo.fechaCosecha!,
            Icons.grass,
            Colors.amber,
            isCompleted: cultivo.esCosechado,
          ),
        ],

        if (cultivo.esEnRiesgo)
          _buildProgressItem(
            'Riesgo Detectado',
            cultivo.fechaRiesgo ?? cultivo.fechaSiembra,
            Icons.warning,
            Colors.orange,
            isCompleted: true,
          ),
      ],
    );
  }

  Widget _buildProgressItem(
    String title,
    String date,
    IconData icon,
    Color color, {
    bool isCompleted = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: isCompleted ? color : color.withOpacity(0.3),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              icon,
              size: 16,
              color: isCompleted ? Colors.white : color,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: isCompleted ? Colors.black87 : Colors.grey[600],
                  ),
                ),
                Text(
                  date,
                  style: TextStyle(
                    fontSize: 11,
                    color: isCompleted ? Colors.black54 : Colors.grey[500],
                  ),
                ),
              ],
            ),
          ),
          if (isCompleted) Icon(Icons.check_circle, color: color, size: 16),
        ],
      ),
    );
  }

  void _mostrarCronograma(Cultivo cultivo) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => CronogramaPage(cultivo: cultivo)),
    );
  }

  List<Cultivo> _ordenarCultivosPorEstado(List<Cultivo> cultivos) {
    // Orden de prioridad: Activos -> En Riesgo -> Cosechados -> Inactivos
    final ordenEstados = {
      Cultivo.ESTADO_ACTIVO: 1,
      Cultivo.ESTADO_EN_RIESGO: 2,
      Cultivo.ESTADO_COSECHADO: 3,
      Cultivo.ESTADO_INACTIVO: 4,
    };

    return List<Cultivo>.from(cultivos)..sort((a, b) {
      final ordenA = ordenEstados[a.estado] ?? 999;
      final ordenB = ordenEstados[b.estado] ?? 999;
      return ordenA.compareTo(ordenB);
    });
  }
}
