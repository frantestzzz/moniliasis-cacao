import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/classes/cronograma_actividad.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:flutter_pdfview/flutter_pdfview.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class CronogramaPage extends StatefulWidget {
  final Cultivo cultivo;

  const CronogramaPage({Key? key, required this.cultivo}) : super(key: key);

  @override
  State<CronogramaPage> createState() => _CronogramaPageState();
}

class _CronogramaPageState extends State<CronogramaPage> {
  late Future<List<CronogramaActividad>> _actividadesFuture;
  List<CronogramaActividad> _actividades = [];
  List<CronogramaActividad> _filteredActividades = [];
  String _selectedFilter = 'todas';
  String? _tipoCultivoNombre;

  @override
  void initState() {
    super.initState();
    _loadActividades();
    _loadTipoCultivo();
  }

  Future<void> _loadActividades() async {
    setState(() {
      _actividadesFuture = _getActividades();
    });
  }

  // 🆕 Cargar el nombre del tipo de cultivo
  Future<void> _loadTipoCultivo() async {
    if (widget.cultivo.tipoId != null) {
      try {
        final tipos = await BasedatoHelper.instance.getAllTiposCultivo();
        final tipo = tipos.firstWhere(
          (t) => t['id'] == widget.cultivo.tipoId,
          orElse: () => {'nombre': 'Desconocido'},
        );
        setState(() {
          _tipoCultivoNombre = (tipo['nombre'] as String).toLowerCase();
        });
      } catch (e) {
        print('Error al cargar tipo de cultivo: $e');
      }
    }
  }

  Future<List<CronogramaActividad>> _getActividades() async {
    try {
      final rows = await BasedatoHelper.instance
          .getCronogramaActividadesPorCultivo(widget.cultivo.id!);
      final actividades = rows
          .map((row) => CronogramaActividad.fromMap(row))
          .toList();
      setState(() {
        _actividades = actividades;
        _applyFilter(_selectedFilter);
      });
      return actividades;
    } catch (e) {
      print('Error al cargar actividades: $e');
      return [];
    }
  }

  void _applyFilter(String filter) {
    setState(() {
      _selectedFilter = filter;
      switch (filter) {
        case 'pendientes':
          _filteredActividades = _actividades
              .where((a) => a.estado == 'pendiente')
              .toList();
          break;
        case 'en_progreso':
          _filteredActividades = _actividades
              .where((a) => a.estado == 'en_progreso')
              .toList();
          break;
        case 'completadas':
          _filteredActividades = _actividades
              .where((a) => a.estado == 'completada')
              .toList();
          break;
        case 'canceladas':
          _filteredActividades = _actividades
              .where((a) => a.estado == 'cancelada')
              .toList();
          break;
        default:
          _filteredActividades = _actividades;
      }
    });
  }

  // 🆕 Mapeo mejorado de tipos de cultivo a guías PDF
  String _getGuiaPath(String tipoCultivoNombre) {
    // Mapeo normalizado de nombres de cultivo a archivos PDF
    final Map<String, String> mapeoGuias = {
      // Plátano
      'plátano': 'platano_guia.pdf',
      'platano': 'platano_guia.pdf',
      'banana': 'platano_guia.pdf',

      // Maíz
      'maíz': 'maiz_guia.pdf',
      'maiz': 'maiz_guia.pdf',
      'choclo': 'maiz_guia.pdf',

      // Arroz
      'arroz': 'arroz_guia.pdf',

      // Café
      'café': 'cafe_guia.pdf',
      'cafe': 'cafe_guia.pdf',
      'coffe': 'cafe_guia.pdf',

      // Tomate
      'tomate': 'tomate_guia.pdf',
      'jitomate': 'tomate_guia.pdf',

      // Cacao
      'cacao': 'cacao_guia.pdf',

      // Yuca
      'yuca': 'yuca_guia.pdf',
      'mandioca': 'yuca_guia.pdf',
      'cassava': 'yuca_guia.pdf',

      // Fríjol
      'fríjol': 'frijol_guia.pdf',
      'frijol': 'frijol_guia.pdf',
      'bean': 'frijol_guia.pdf',

      // Papa
      'papa': 'papa_guia.pdf',
      'patata': 'papa_guia.pdf',
      'potato': 'papa_guia.pdf',
    };

    // Normalizar el nombre del cultivo (quitar tildes, convertir a minúsculas)
    final nombreNormalizado = tipoCultivoNombre
        .toLowerCase()
        .replaceAll('á', 'a')
        .replaceAll('é', 'e')
        .replaceAll('í', 'i')
        .replaceAll('ó', 'o')
        .replaceAll('ú', 'u')
        .replaceAll('ñ', 'n');

    // Buscar coincidencia exacta
    if (mapeoGuias.containsKey(nombreNormalizado)) {
      return 'assets/guias/${mapeoGuias[nombreNormalizado]}';
    }

    // Buscar coincidencia parcial (si el nombre contiene el tipo)
    for (final entry in mapeoGuias.entries) {
      if (nombreNormalizado.contains(entry.key)) {
        return 'assets/guias/${entry.value}';
      }
    }

    // Si no encuentra coincidencia, retornar null
    return '';
  }

  // 🆕 Abrir la guía PDF
  Future<void> _abrirGuiaPDF() async {
    if (_tipoCultivoNombre == null || _tipoCultivoNombre!.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No se pudo determinar el tipo de cultivo'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    // Obtener la ruta de la guía usando el mapeo mejorado
    final guiaPath = _getGuiaPath(_tipoCultivoNombre!);

    if (guiaPath.isEmpty) {
      // Mostrar diálogo de guía no disponible con más información
      if (mounted) {
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            title: Row(
              children: [
                Icon(Icons.info_outline, color: Colors.orange[700], size: 28),
                const SizedBox(width: 12),
                const Text('Guía no disponible'),
              ],
            ),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'No hay una guía específica disponible para:',
                  style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                ),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _tipoCultivoNombre!.toUpperCase(),
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Cultivos con guía disponible:',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 4,
                  children:
                      [
                            'Plátano',
                            'Maíz',
                            'Arroz',
                            'Café',
                            'Tomate',
                            'Cacao',
                            'Yuca',
                            'Fríjol',
                            'Papa',
                          ]
                          .map(
                            (cultivo) => Chip(
                              label: Text(
                                cultivo,
                                style: const TextStyle(fontSize: 12),
                              ),
                              backgroundColor: Colors.green[50],
                              side: BorderSide(color: Colors.green[200]!),
                            ),
                          )
                          .toList(),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Entendido'),
              ),
            ],
          ),
        );
      }
      return;
    }

    // Verificar si existe el archivo
    try {
      await rootBundle.load(guiaPath);

      // Si existe, navegar al visor
      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => PDFViewerPage(
              assetPath: guiaPath,
              tipoCultivo: _tipoCultivoNombre!,
            ),
          ),
        );
      }
    } catch (e) {
      // Si el archivo no existe, mostrar error
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error al cargar la guía: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Cronograma - ${widget.cultivo.nombre}'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        actions: [
          // 🆕 BOTÓN DE GUÍA EN EL APPBAR
          IconButton(
            icon: const Icon(Icons.menu_book),
            onPressed: _abrirGuiaPDF,
            tooltip: 'Ver Guía del Cultivo',
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadActividades,
        child: Column(
          children: [
            // Resumen del cultivo
            _buildCultivoSummary(),

            // 🆕 BOTÓN DE GUÍA (encima del FAB)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _abrirGuiaPDF,
                  icon: const Icon(Icons.menu_book, size: 20),
                  label: const Text(
                    'Ver Guía del Cultivo',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue[700],
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 3,
                  ),
                ),
              ),
            ),

            // Lista de actividades
            Expanded(
              child: FutureBuilder<List<CronogramaActividad>>(
                future: _actividadesFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  }

                  if (snapshot.hasError) {
                    return Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.error, color: Colors.red, size: 64),
                          const SizedBox(height: 16),
                          Text('Error: ${snapshot.error}'),
                          const SizedBox(height: 16),
                          ElevatedButton(
                            onPressed: _loadActividades,
                            child: const Text('Reintentar'),
                          ),
                        ],
                      ),
                    );
                  }

                  final actividades = snapshot.data ?? [];

                  if (actividades.isEmpty) {
                    return _buildEmptyState();
                  }

                  if (_filteredActividades.isEmpty &&
                      _selectedFilter != 'todas') {
                    return _buildFilteredEmptyState();
                  }

                  return ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _filteredActividades.length,
                    itemBuilder: (context, index) {
                      final actividad = _filteredActividades[index];
                      return _buildActividadCard(actividad);
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showActividadForm(),
        backgroundColor: Colors.green,
        child: const Icon(Icons.add),
        tooltip: 'Agregar Actividad',
      ),
    );
  }

  Widget _buildCultivoSummary() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.green[400]!, Colors.green[600]!],
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.green.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.agriculture, color: Colors.white, size: 24),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.cultivo.nombre,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Área: ${widget.cultivo.area.toStringAsFixed(1)} ha',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Interactive filter boxes
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                _buildFilterCard(
                  'Todas',
                  '${_actividades.length}',
                  Icons.task_alt,
                  const Color.fromARGB(255, 6, 6, 6),
                  const Color.fromARGB(255, 220, 221, 237)!,
                  'todas',
                ),
                const SizedBox(width: 12),
                _buildFilterCard(
                  'Pendientes',
                  '${_actividades.where((a) => a.estado == 'pendiente').length}',
                  Icons.pending,
                  Colors.orange,
                  Colors.white,
                  'pendientes',
                ),
                const SizedBox(width: 12),
                _buildFilterCard(
                  'En Progreso',
                  '${_actividades.where((a) => a.estado == 'en_progreso').length}',
                  Icons.hourglass_empty,
                  Colors.blue,
                  Colors.white,
                  'en_progreso',
                ),
                const SizedBox(width: 12),
                _buildFilterCard(
                  'Completadas',
                  '${_actividades.where((a) => a.estado == 'completada').length}',
                  Icons.check_circle,
                  const Color.fromARGB(255, 7, 146, 37),
                  Colors.white,
                  'completadas',
                ),
                const SizedBox(width: 12),
                _buildFilterCard(
                  'Canceladas',
                  '${_actividades.where((a) => a.estado == 'cancelada').length}',
                  Icons.cancel,
                  Colors.red,
                  Colors.white,
                  'canceladas',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFilterCard(
    String title,
    String count,
    IconData icon,
    Color bgColor,
    Color textColor,
    String filterKey,
  ) {
    final bool isActive = _selectedFilter == filterKey;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => _applyFilter(filterKey),
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: isActive ? textColor : bgColor,
            borderRadius: BorderRadius.circular(12),
            border: isActive
                ? Border.all(color: textColor, width: 2)
                : Border.all(color: textColor.withOpacity(0.3)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: isActive ? bgColor : textColor, size: 20),
              const SizedBox(height: 4),
              Text(
                count,
                style: TextStyle(
                  color: isActive ? bgColor : textColor,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                title,
                style: TextStyle(
                  color: isActive ? bgColor : textColor,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.schedule, size: 64, color: Colors.grey[400]),
          const SizedBox(height: 16),
          Text(
            'No hay actividades programadas',
            style: TextStyle(
              fontSize: 18,
              color: Colors.grey[600],
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Agrega tu primera actividad para comenzar',
            style: TextStyle(fontSize: 14, color: Colors.grey[500]),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () => _showActividadForm(),
            icon: const Icon(Icons.add),
            label: const Text('Agregar Actividad'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFilteredEmptyState() {
    String filterTitle = '';
    IconData filterIcon = Icons.filter_list;

    switch (_selectedFilter) {
      case 'pendientes':
        filterTitle = 'Pendientes';
        filterIcon = Icons.pending;
        break;
      case 'en_progreso':
        filterTitle = 'En Progreso';
        filterIcon = Icons.hourglass_empty;
        break;
      case 'completadas':
        filterTitle = 'Completadas';
        filterIcon = Icons.check_circle;
        break;
      case 'canceladas':
        filterTitle = 'Canceladas';
        filterIcon = Icons.cancel;
        break;
    }

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(filterIcon, size: 64, color: Colors.grey[400]),
          const SizedBox(height: 16),
          Text(
            'No hay actividades $filterTitle',
            style: TextStyle(
              fontSize: 18,
              color: Colors.grey[600],
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Intenta con otro filtro o agrega una nueva actividad',
            style: TextStyle(fontSize: 14, color: Colors.grey[500]),
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                onPressed: () => _applyFilter('todas'),
                icon: const Icon(Icons.list),
                label: const Text('Ver Todas'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.grey,
                  foregroundColor: Colors.white,
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton.icon(
                onPressed: () => _showActividadForm(),
                icon: const Icon(Icons.add),
                label: const Text('Agregar Actividad'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildActividadCard(CronogramaActividad actividad) {
    final color = CronogramaActividad.getColorEstado(actividad.estado);
    final icon = CronogramaActividad.getIconoTipo(actividad.tipo);
    final tipoNombre = CronogramaActividad.getTipoNombre(actividad.tipo);
    final estadoNombre = CronogramaActividad.getEstadoNombre(actividad.estado);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 3,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: color.withOpacity(0.3), width: 1),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(icon, color: color, size: 20),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        actividad.titulo,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        tipoNombre,
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: color.withOpacity(0.3)),
                  ),
                  child: Text(
                    estadoNombre,
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      color: color,
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            // Description
            if (actividad.descripcion.isNotEmpty) ...[
              Text(
                actividad.descripcion,
                style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 8),
            ],

            // Details row
            Row(
              children: [
                Icon(Icons.calendar_today, size: 16, color: Colors.grey[600]),
                const SizedBox(width: 4),
                Text(
                  'Programada: ${_formatDate(actividad.fechaProgramada)}',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
                if (actividad.fechaRealizada != null) ...[
                  const SizedBox(width: 16),
                  Icon(Icons.check_circle, size: 16, color: Colors.green),
                  const SizedBox(width: 4),
                  Text(
                    'Realizada: ${_formatDate(actividad.fechaRealizada!)}',
                    style: TextStyle(fontSize: 12, color: Colors.green[700]),
                  ),
                ],
              ],
            ),

            if (actividad.costo != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.attach_money, size: 16, color: Colors.grey[600]),
                  const SizedBox(width: 4),
                  Text(
                    'Costo: \$${actividad.costo!.toStringAsFixed(2)}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ],

            // Actions
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                if (actividad.estado != 'completada')
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ElevatedButton.icon(
                        onPressed: () => _confirmarCompletar(actividad),
                        icon: const Icon(Icons.check, size: 18),
                        label: const Text('Completar'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 10,
                          ),
                          textStyle: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                  ),
                IconButton(
                  icon: const Icon(Icons.edit, color: Colors.blue),
                  onPressed: () => _showActividadForm(actividad: actividad),
                  tooltip: 'Editar',
                ),
                IconButton(
                  icon: const Icon(Icons.delete, color: Colors.red),
                  onPressed: () => _eliminarActividad(actividad),
                  tooltip: 'Eliminar',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _formatDate(String dateString) {
    try {
      final date = DateTime.parse(dateString);
      return DateFormat('dd/MM/yyyy').format(date);
    } catch (e) {
      return dateString;
    }
  }

  void _showActividadForm({CronogramaActividad? actividad}) {
    showDialog(
      context: context,
      builder: (context) => ActividadFormDialog(
        cultivo: widget.cultivo,
        actividad: actividad,
        onSave: () {
          _loadActividades();
          Navigator.pop(context);
        },
      ),
    );
  }

  void _confirmarCompletar(CronogramaActividad actividad) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.check_circle, color: Colors.green, size: 24),
            const SizedBox(width: 12),
            const Text('Marcar como Completado'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '¿Estás seguro de que quieres marcar esta actividad como completada?',
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    actividad.titulo,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    CronogramaActividad.getTipoNombre(actividad.tipo),
                    style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Fecha de realización: ${_formatDate(DateTime.now().toIso8601String().split('T')[0])}',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[600],
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('No'),
            style: TextButton.styleFrom(foregroundColor: Colors.grey[600]),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _marcarCompletada(actividad);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
            ),
            child: const Text('Sí, Completar'),
          ),
        ],
      ),
    );
  }

  void _marcarCompletada(CronogramaActividad actividad) async {
    try {
      await BasedatoHelper.instance.marcarCronogramaActividadCompletada(
        actividad.id!,
      );
      _loadActividades();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Actividad marcada como completada'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error al marcar actividad: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _eliminarActividad(CronogramaActividad actividad) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Eliminar Actividad'),
        content: Text(
          '¿Estás seguro de que quieres eliminar "${actividad.titulo}"?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () async {
              try {
                await BasedatoHelper.instance.eliminarCronogramaActividad(
                  actividad.id!,
                );
                _loadActividades();
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Actividad eliminada'),
                    backgroundColor: Colors.green,
                  ),
                );
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('Error al eliminar actividad: $e'),
                    backgroundColor: Colors.red,
                  ),
                );
              }
            },
            child: const Text('Eliminar', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}

// 🆕 VISOR DE PDF
class PDFViewerPage extends StatefulWidget {
  final String assetPath;
  final String tipoCultivo;

  const PDFViewerPage({
    Key? key,
    required this.assetPath,
    required this.tipoCultivo,
  }) : super(key: key);

  @override
  State<PDFViewerPage> createState() => _PDFViewerPageState();
}

class _PDFViewerPageState extends State<PDFViewerPage> {
  String? _localPath;
  bool _isLoading = true;
  String? _error;
  int _currentPage = 0;
  int _totalPages = 0;

  @override
  void initState() {
    super.initState();
    _loadPDF();
  }

  Future<void> _loadPDF() async {
    try {
      final bytes = await rootBundle.load(widget.assetPath);
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/temp_guia.pdf');
      await file.writeAsBytes(bytes.buffer.asUint8List());

      setState(() {
        _localPath = file.path;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Guía - ${widget.tipoCultivo.toUpperCase()}'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        actions: [
          if (_totalPages > 0)
            Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(
                  'Página ${_currentPage + 1} de $_totalPages',
                  style: const TextStyle(fontSize: 14),
                ),
              ),
            ),
        ],
      ),
      body: _isLoading
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Cargando guía...'),
                ],
              ),
            )
          : _error != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.error, color: Colors.red, size: 64),
                  SizedBox(height: 16),
                  Text('Error al cargar la guía'),
                  SizedBox(height: 8),
                  Text(
                    _error!,
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            )
          : PDFView(
              filePath: _localPath!,
              enableSwipe: true,
              swipeHorizontal: false,
              autoSpacing: false,
              pageFling: true,
              pageSnap: true,
              defaultPage: 0,
              fitPolicy: FitPolicy.BOTH,
              preventLinkNavigation: false,
              onRender: (pages) {
                setState(() {
                  _totalPages = pages ?? 0;
                });
              },
              onError: (error) {
                print('Error al renderizar PDF: $error');
              },
              onPageError: (page, error) {
                print('Error en página $page: $error');
              },
              onViewCreated: (PDFViewController controller) {
                // Opcional: guardar el controlador si necesitas control programático
              },
              onPageChanged: (int? page, int? total) {
                setState(() {
                  _currentPage = page ?? 0;
                  _totalPages = total ?? 0;
                });
              },
            ),
    );
  }
}

// Dialog del formulario (sin cambios significativos, solo para completitud)
class ActividadFormDialog extends StatefulWidget {
  final Cultivo cultivo;
  final CronogramaActividad? actividad;
  final VoidCallback onSave;

  const ActividadFormDialog({
    Key? key,
    required this.cultivo,
    this.actividad,
    required this.onSave,
  }) : super(key: key);

  @override
  State<ActividadFormDialog> createState() => _ActividadFormDialogState();
}

class _ActividadFormDialogState extends State<ActividadFormDialog> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _tituloCtrl;
  late TextEditingController _descripcionCtrl;
  late TextEditingController _fechaCtrl;
  late TextEditingController _costoCtrl;
  late TextEditingController _notasCtrl;
  late String _tipoSeleccionado;
  late String _estadoSeleccionado;

  @override
  void initState() {
    super.initState();
    _tituloCtrl = TextEditingController(text: widget.actividad?.titulo ?? '');
    _descripcionCtrl = TextEditingController(
      text: widget.actividad?.descripcion ?? '',
    );
    _fechaCtrl = TextEditingController(
      text:
          widget.actividad?.fechaProgramada ??
          DateTime.now().toIso8601String().split('T')[0],
    );
    _costoCtrl = TextEditingController(
      text: widget.actividad?.costo?.toString() ?? '',
    );
    _notasCtrl = TextEditingController(text: widget.actividad?.notas ?? '');
    _tipoSeleccionado = widget.actividad?.tipo ?? 'otro';
    _estadoSeleccionado = widget.actividad?.estado ?? 'pendiente';
  }

  @override
  void dispose() {
    _tituloCtrl.dispose();
    _descripcionCtrl.dispose();
    _fechaCtrl.dispose();
    _costoCtrl.dispose();
    _notasCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: EdgeInsets.zero, // Elimina márgenes por defecto
      child: Container(
        width: MediaQuery.of(context).size.width, // 100% del ancho
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.85,
        ),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 20,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Header con gradiente
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [Colors.green[600]!, Colors.green[400]!],
                ),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(20),
                  topRight: Radius.circular(20),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Icon(
                          widget.actividad == null
                              ? Icons.add_task
                              : Icons.edit_calendar,
                          color: Colors.white,
                          size: 28,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              widget.actividad == null
                                  ? 'Nueva Actividad'
                                  : 'Editar Actividad',
                              style: const TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'Cultivo: ${widget.cultivo.nombre}',
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.white.withOpacity(0.9),
                              ),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.close, color: Colors.white),
                        style: IconButton.styleFrom(
                          backgroundColor: Colors.white.withOpacity(0.2),
                          padding: const EdgeInsets.all(8),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // Content
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Sección: Información Básica
                      _buildSectionHeader(
                        'Información Básica',
                        Icons.info_outline,
                      ),
                      const SizedBox(height: 16),

                      // Campo de título más grande
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.grey[50],
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.grey[200]!),
                        ),
                        child: TextFormField(
                          controller: _tituloCtrl,
                          decoration: const InputDecoration(
                            labelText: 'Título de la actividad',
                            hintText: 'Ej: Fertilización de maíz',
                            prefixIcon: Icon(Icons.title, color: Colors.green),
                            border: InputBorder.none,
                            contentPadding: EdgeInsets.all(16),
                          ),
                          style: const TextStyle(fontSize: 16),
                          validator: (v) => (v == null || v.isEmpty)
                              ? 'Este campo es requerido'
                              : null,
                        ),
                      ),
                      const SizedBox(height: 20),

                      // Campo de descripción más grande
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.grey[50],
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.grey[200]!),
                        ),
                        child: TextFormField(
                          controller: _descripcionCtrl,
                          decoration: const InputDecoration(
                            labelText: 'Descripción detallada',
                            hintText:
                                'Describe qué vas a hacer, cómo lo harás y los detalles importantes...',
                            prefixIcon: Icon(
                              Icons.description,
                              color: Colors.green,
                            ),
                            border: InputBorder.none,
                            contentPadding: EdgeInsets.all(16),
                          ),
                          maxLines: 4,
                          style: const TextStyle(fontSize: 16),
                          validator: (v) => (v == null || v.isEmpty)
                              ? 'Este campo es requerido'
                              : null,
                        ),
                      ),
                      const SizedBox(height: 24),

                      // Sección: Configuración
                      _buildSectionHeader('Configuración', Icons.settings),
                      const SizedBox(height: 16),

                      // Tipo y Fecha en columna
                      Column(
                        children: [
                          // Campo de tipo
                          Container(
                            width: double.infinity,
                            decoration: BoxDecoration(
                              color: Colors.grey[50],
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.grey[200]!),
                            ),
                            child: DropdownButtonFormField<String>(
                              value: _tipoSeleccionado,
                              decoration: const InputDecoration(
                                labelText: 'Tipo de actividad',
                                prefixIcon: Icon(
                                  Icons.category,
                                  color: Colors.green,
                                ),
                                border: InputBorder.none,
                                contentPadding: EdgeInsets.all(16),
                              ),
                              items: CronogramaActividad.TIPOS_ACTIVIDAD.map((
                                tipo,
                              ) {
                                return DropdownMenuItem(
                                  value: tipo,
                                  child: Text(
                                    CronogramaActividad.getTipoNombre(tipo),
                                  ),
                                );
                              }).toList(),
                              onChanged: (v) =>
                                  setState(() => _tipoSeleccionado = v!),
                            ),
                          ),
                          const SizedBox(height: 16),
                          // Campo de fecha
                          Container(
                            width: double.infinity,
                            decoration: BoxDecoration(
                              color: Colors.grey[50],
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.grey[200]!),
                            ),
                            child: TextFormField(
                              controller: _fechaCtrl,
                              decoration: const InputDecoration(
                                labelText: 'Fecha programada',
                                prefixIcon: Icon(
                                  Icons.calendar_today,
                                  color: Colors.green,
                                ),
                                border: InputBorder.none,
                                contentPadding: EdgeInsets.all(16),
                              ),
                              readOnly: true,
                              onTap: () async {
                                final date = await showDatePicker(
                                  context: context,
                                  initialDate:
                                      DateTime.tryParse(_fechaCtrl.text) ??
                                      DateTime.now(),
                                  firstDate: DateTime.now(),
                                  lastDate: DateTime(2030),
                                  builder: (context, child) {
                                    return Theme(
                                      data: ThemeData.light().copyWith(
                                        colorScheme: ColorScheme.fromSeed(
                                          seedColor: Colors.green,
                                          primary: Colors.green,
                                        ),
                                      ),
                                      child: child!,
                                    );
                                  },
                                );
                                if (date != null) {
                                  _fechaCtrl.text = date
                                      .toIso8601String()
                                      .split('T')[0];
                                }
                              },
                              validator: (v) => (v == null || v.isEmpty)
                                  ? 'Este campo es requerido'
                                  : null,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // Footer con botones
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.grey[50],
                borderRadius: const BorderRadius.only(
                  bottomLeft: Radius.circular(20),
                  bottomRight: Radius.circular(20),
                ),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => Navigator.pop(context),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        side: const BorderSide(color: Colors.grey),
                      ),
                      child: const Text(
                        'Cancelar',
                        style: TextStyle(fontSize: 16, color: Colors.grey),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    flex: 2,
                    child: ElevatedButton(
                      onPressed: () async {
                        if (_formKey.currentState!.validate()) {
                          final actividad = CronogramaActividad(
                            id: widget.actividad?.id,
                            cultivoId: widget.cultivo.id!,
                            titulo: _tituloCtrl.text,
                            descripcion: _descripcionCtrl.text,
                            fechaProgramada: _fechaCtrl.text,
                            tipo: _tipoSeleccionado,
                            estado: _estadoSeleccionado,
                            costo: _costoCtrl.text.isNotEmpty
                                ? double.parse(_costoCtrl.text)
                                : null,
                            notas: _notasCtrl.text.isNotEmpty
                                ? _notasCtrl.text
                                : null,
                            creadoEn: widget.actividad?.creadoEn,
                          );

                          if (widget.actividad == null) {
                            await BasedatoHelper.instance
                                .insertarCronogramaActividad(actividad.toMap());
                          } else {
                            await BasedatoHelper.instance
                                .actualizarCronogramaActividad(
                                  widget.actividad!.id!,
                                  actividad.toMap(),
                                );
                          }

                          widget.onSave();
                        }
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 3,
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            widget.actividad == null ? Icons.add : Icons.save,
                            size: 20,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            widget.actividad == null
                                ? 'Agregar Actividad'
                                : 'Guardar Cambios',
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.green.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: Colors.green, size: 20),
        ),
        const SizedBox(width: 12),
        Text(
          title,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.black87,
          ),
        ),
      ],
    );
  }
}
