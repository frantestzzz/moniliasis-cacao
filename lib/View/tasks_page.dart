import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:amgeca/classes/cronograma_actividad.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class TasksPage extends StatefulWidget {
  const TasksPage({Key? key}) : super(key: key);

  @override
  State<TasksPage> createState() => _TasksPageState();
}

class _TasksPageState extends State<TasksPage> {
  late Future<List<Map<String, dynamic>>> _tasksFuture;

  @override
  void initState() {
    super.initState();
    _loadTasks();
  }

  Future<void> _loadTasks() async {
    setState(() {
      _tasksFuture = _getTasksFromCronograma();
    });
  }

  Future<List<Map<String, dynamic>>> _getTasksFromCronograma() async {
    try {
      // Obtener fecha actual y mañana
      final now = DateTime.now();
      final today = DateTime(now.year, now.month, now.day);
      final tomorrow = today.add(const Duration(days: 1));

      // Formato YYYY-MM-DD para comparación
      final todayStr = today.toIso8601String().split('T')[0];
      final tomorrowStr = tomorrow.toIso8601String().split('T')[0];

      // Obtener todas las actividades pendientes de hoy y mañana
      final allCultivos = await BasedatoHelper.instance.getAllCultivos();
      List<Map<String, dynamic>> tasks = [];

      for (final cultivoRow in allCultivos) {
        final cultivo = Cultivo.fromMap(cultivoRow);
        final actividades = await BasedatoHelper.instance
            .getCronogramaActividadesPorCultivo(cultivo.id!);

        for (final actividadRow in actividades) {
          final actividad = CronogramaActividad.fromMap(actividadRow);

          // Solo incluir actividades pendientes o en progreso
          if (actividad.estado == 'pendiente' ||
              actividad.estado == 'en_progreso') {
            // Verificar si la fecha es hoy o mañana
            if (actividad.fechaProgramada == todayStr ||
                actividad.fechaProgramada == tomorrowStr) {
              tasks.add({
                'actividad': actividad,
                'cultivo': cultivo,
                'isToday': actividad.fechaProgramada == todayStr,
                'isTomorrow': actividad.fechaProgramada == tomorrowStr,
              });
            }
          }
        }
      }

      // Ordenar: primero las de hoy, luego las de mañana
      tasks.sort((a, b) {
        if (a['isToday'] && !b['isToday']) return -1;
        if (!a['isToday'] && b['isToday']) return 1;
        return 0;
      });

      return tasks;
    } catch (e) {
      print('Error al cargar tareas: $e');
      return [];
    }
  }

  Future<void> _markAsCompleted(
    CronogramaActividad actividad,
    Cultivo cultivo,
  ) async {
    try {
      await BasedatoHelper.instance.marcarCronogramaActividadCompletada(
        actividad.id!,
      );

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('✅ Actividad "${actividad.titulo}" completada'),
          backgroundColor: Colors.green,
          duration: const Duration(seconds: 2),
        ),
      );

      // Recargar tareas
      _loadTasks();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('❌ Error al completar actividad: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _confirmarCompletar(CronogramaActividad actividad, Cultivo cultivo) {
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
                    '${cultivo.nombre} • ${CronogramaActividad.getTipoNombre(actividad.tipo)}',
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
              _markAsCompleted(actividad, cultivo);
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tareas de Hoy y Mañana'),
        backgroundColor: Colors.green[800],
        foregroundColor: Colors.white,
      ),
      body: RefreshIndicator(
        onRefresh: _loadTasks,
        child: FutureBuilder<List<Map<String, dynamic>>>(
          future: _tasksFuture,
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
                      onPressed: _loadTasks,
                      child: const Text('Reintentar'),
                    ),
                  ],
                ),
              );
            }

            final tasks = snapshot.data ?? [];

            if (tasks.isEmpty) {
              return _buildEmptyState();
            }

            return Column(
              children: [
                // Resumen de tareas
                _buildTasksSummary(tasks),

                // Lista de tareas
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: tasks.length,
                    itemBuilder: (context, index) {
                      final taskData = tasks[index];
                      final actividad =
                          taskData['actividad'] as CronogramaActividad;
                      final cultivo = taskData['cultivo'] as Cultivo;
                      final isToday = taskData['isToday'] as bool;
                      final isTomorrow = taskData['isTomorrow'] as bool;

                      return _buildTaskCard(
                        actividad,
                        cultivo,
                        isToday,
                        isTomorrow,
                      );
                    },
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildTasksSummary(List<Map<String, dynamic>> tasks) {
    final todayTasks = tasks.where((t) => t['isToday']).length;
    final tomorrowTasks = tasks.where((t) => t['isTomorrow']).length;

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.green[600]!, Colors.green[800]!],
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
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildSummaryItem('Hoy', todayTasks.toString(), Icons.today, true),
          _buildSummaryItem(
            'Mañana',
            tomorrowTasks.toString(),
            Icons.event,
            false,
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryItem(
    String label,
    String count,
    IconData icon,
    bool isActive,
  ) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: isActive ? Colors.white : Colors.white.withOpacity(0.2),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(
            icon,
            color: isActive ? Colors.green[800] : Colors.white,
            size: 24,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          count,
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label,
          style: TextStyle(color: Colors.white.withOpacity(0.9), fontSize: 12),
        ),
      ],
    );
  }

  Widget _buildTaskCard(
    CronogramaActividad actividad,
    Cultivo cultivo,
    bool isToday,
    bool isTomorrow,
  ) {
    final color = CronogramaActividad.getColorEstado(actividad.estado);
    final tipoNombre = CronogramaActividad.getTipoNombre(actividad.tipo);

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
            // Header con fecha
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: isToday ? Colors.orange[100] : Colors.blue[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    isToday ? Icons.today : Icons.event,
                    color: isToday ? Colors.orange : Colors.blue,
                    size: 20,
                  ),
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
                        '${cultivo.nombre} • $tipoNombre',
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
                    color: isToday ? Colors.orange : Colors.blue,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    isToday ? 'Hoy' : 'Mañana',
                    style: const TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            // Descripción
            if (actividad.descripcion.isNotEmpty) ...[
              Text(
                actividad.descripcion,
                style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 8),
            ],

            // Fecha programada
            Row(
              children: [
                Icon(Icons.calendar_today, size: 16, color: Colors.grey[600]),
                const SizedBox(width: 4),
                Text(
                  'Programada: ${_formatDate(actividad.fechaProgramada)}',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
              ],
            ),

            const SizedBox(height: 12),

            // Botón de completar
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: () => _confirmarCompletar(actividad, cultivo),
                icon: const Icon(Icons.check, size: 18),
                label: const Text('Marcar como Completada'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
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
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(32),
            decoration: BoxDecoration(
              color: Colors.green[50],
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.task_alt, size: 64, color: Colors.green[700]),
          ),
          const SizedBox(height: 24),
          Text(
            '¡No hay tareas para hoy ni mañana!',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.grey[800],
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Todas las actividades están al día',
            style: TextStyle(fontSize: 14, color: Colors.grey[600]),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: _loadTasks,
            icon: const Icon(Icons.refresh),
            label: const Text('Actualizar'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
            ),
          ),
        ],
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
}
