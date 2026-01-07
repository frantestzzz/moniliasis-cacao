import 'package:flutter/material.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class AlertsPage extends StatefulWidget {
  const AlertsPage({Key? key}) : super(key: key);

  @override
  State<AlertsPage> createState() => _AlertsPageState();
}

class _AlertsPageState extends State<AlertsPage> {
  late Future<List<Cultivo>> _cultivosEnRiesgoFuture;

  @override
  void initState() {
    super.initState();
    _loadCultivosEnRiesgo();
  }

  Future<void> _loadCultivosEnRiesgo() async {
    setState(() {
      _cultivosEnRiesgoFuture = _getCultivosEnRiesgo();
    });
  }

  Future<List<Cultivo>> _getCultivosEnRiesgo() async {
    try {
      final allCultivos = await BasedatoHelper.instance.getAllCultivos();
      List<Cultivo> cultivosEnRiesgo = [];

      for (final cultivoRow in allCultivos) {
        final cultivo = Cultivo.fromMap(cultivoRow);
        if (cultivo.esEnRiesgo) {
          cultivosEnRiesgo.add(cultivo);
        }
      }

      return cultivosEnRiesgo;
    } catch (e) {
      print('Error al cargar cultivos en riesgo: $e');
      return [];
    }
  }

  String _getSeverityLevel(Cultivo cultivo) {
    // Determinar nivel de severidad basado en información del cultivo
    if (cultivo.fechaCosecha != null) {
      final fechaCosecha = DateTime.tryParse(cultivo.fechaCosecha!);
      if (fechaCosecha != null) {
        final diasParaCosecha = fechaCosecha.difference(DateTime.now()).inDays;

        if (diasParaCosecha <= 7) {
          return 'critical'; // Cercano a cosecha
        } else if (diasParaCosecha <= 30) {
          return 'warning'; // Próximo a cosecha
        }
      }
    }
    return 'warning'; // Por defecto
  }

  String _getAlertMessage(Cultivo cultivo) {
    List<String> mensajes = [];

    // Analizar las notas para detectar problemas específicos
    if (cultivo.notas != null && cultivo.notas!.isNotEmpty) {
      final notas = cultivo.notas!.toLowerCase();

      // Detectar plagas
      if (notas.contains('plaga') ||
          notas.contains('insecto') ||
          notas.contains('pulgon')) {
        if (notas.contains('pulgon'))
          mensajes.add('Presencia de pulgones detectada');
        if (notas.contains('mosca')) mensajes.add('Actividad de mosca blanca');
        if (notas.contains('hormiga')) mensajes.add('Ataque de hormigas');
        if (notas.contains('oruga'))
          mensajes.add('Daño por orugas');
        else
          mensajes.add('Plaga detectada en el cultivo');
      }

      // Detectar enfermedades
      if (notas.contains('hongo') ||
          notas.contains('moho') ||
          notas.contains('mancha')) {
        if (notas.contains('moho')) mensajes.add('Síntomas de moho');
        if (notas.contains('mancha'))
          mensajes.add('Manchas foliares detectadas');
        else
          mensajes.add('Posible enfermedad fúngica');
      }

      // Detectar problemas de riego
      if (notas.contains('sequia') ||
          notas.contains('falta agua') ||
          notas.contains('seco')) {
        mensajes.add('Estrés hídrico detectado');
      }
      if (notas.contains('exceso agua') ||
          notas.contains('inundado') ||
          notas.contains('encharcado')) {
        mensajes.add('Saturación de humedad');
      }

      // Detectar problemas nutricionales
      if (notas.contains('amarillo') || notas.contains('clorosis')) {
        mensajes.add('Síntomas de deficiencia nutricional');
      }

      // Detectar problemas de crecimiento
      if (notas.contains('crecimiento lento') || notas.contains('detenido')) {
        mensajes.add('Crecimiento anómalo detectado');
      }
    }

    // Si no se detectaron problemas específicos, usar mensajes por fecha de cosecha
    if (mensajes.isEmpty) {
      if (cultivo.fechaCosecha != null) {
        final fechaCosecha = DateTime.tryParse(cultivo.fechaCosecha!);
        if (fechaCosecha != null) {
          final diasParaCosecha = fechaCosecha
              .difference(DateTime.now())
              .inDays;

          if (diasParaCosecha <= 7) {
            mensajes.add('Cosecha inminente en $diasParaCosecha días');
          } else if (diasParaCosecha <= 30) {
            mensajes.add('Cosecha programada en $diasParaCosecha días');
          } else {
            mensajes.add('Requiere atención especial');
          }
        } else {
          mensajes.add('Requiere atención especial');
        }
      } else {
        mensajes.add('Requiere atención especial');
      }
    }

    // Agregar información adicional de las notas si no se usaron antes
    if (cultivo.notas != null && cultivo.notas!.isNotEmpty) {
      final notas = cultivo.notas!;
      if (!notas.toLowerCase().contains('plaga') &&
          !notas.toLowerCase().contains('hongo') &&
          !notas.toLowerCase().contains('sequia')) {
        mensajes.add('Observación: ${notas}');
      }
    }

    return mensajes.join(' • ');
  }

  Color _severityColor(String severity) {
    switch (severity.toLowerCase()) {
      case 'critical':
        return Colors.red;
      case 'warning':
        return Colors.orange;
      default:
        return Colors.green;
    }
  }

  IconData _severityIcon(String severity) {
    switch (severity.toLowerCase()) {
      case 'critical':
        return Icons.dangerous;
      case 'warning':
        return Icons.warning_amber;
      default:
        return Icons.info;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Alertas de Cultivos en Riesgo'),
        backgroundColor: Colors.green[800],
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _loadCultivosEnRiesgo,
            tooltip: 'Actualizar alertas',
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadCultivosEnRiesgo,
        child: FutureBuilder<List<Cultivo>>(
          future: _cultivosEnRiesgoFuture,
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
                      onPressed: _loadCultivosEnRiesgo,
                      child: const Text('Reintentar'),
                    ),
                  ],
                ),
              );
            }

            final cultivosEnRiesgo = snapshot.data ?? [];

            if (cultivosEnRiesgo.isEmpty) {
              return _buildEmptyState();
            }

            return Column(
              children: [
                // Resumen de alertas
                _buildAlertSummary(cultivosEnRiesgo),

                // Lista de alertas
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: cultivosEnRiesgo.length,
                    itemBuilder: (context, index) {
                      final cultivo = cultivosEnRiesgo[index];
                      final severity = _getSeverityLevel(cultivo);

                      return _buildAlertCard(cultivo, severity);
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

  Widget _buildAlertSummary(List<Cultivo> cultivosEnRiesgo) {
    final criticalCount = cultivosEnRiesgo
        .where((c) => _getSeverityLevel(c) == 'critical')
        .length;
    final warningCount = cultivosEnRiesgo
        .where((c) => _getSeverityLevel(c) == 'warning')
        .length;

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.red[600]!, Colors.orange[600]!],
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.red.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildSummaryItem(
            'Críticas',
            criticalCount.toString(),
            Icons.dangerous,
            Colors.red,
          ),
          _buildSummaryItem(
            'Advertencia',
            warningCount.toString(),
            Icons.warning_amber,
            Colors.orange,
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryItem(
    String label,
    String count,
    IconData icon,
    Color color,
  ) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: color, size: 24),
        ),
        const SizedBox(height: 8),
        Text(
          count,
          style: const TextStyle(
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

  Widget _buildAlertCard(Cultivo cultivo, String severity) {
    final color = _severityColor(severity);
    final icon = _severityIcon(severity);
    final severityLabel = severity == 'critical' ? 'Crítico' : 'Advertencia';
    final message = _getAlertMessage(cultivo);

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
            // Header con severidad
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
                        cultivo.nombre,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        'Área: ${cultivo.area} ha • ${cultivo.tipoSuelo}',
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
                    color: color,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    severityLabel,
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

            // Mensaje de alerta
            Text(
              message,
              style: TextStyle(fontSize: 14, color: Colors.grey[700]),
            ),

            // Información detallada
            if (cultivo.fechaCosecha != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.calendar_today, size: 16, color: Colors.grey[600]),
                  const SizedBox(width: 4),
                  Text(
                    'Cosecha: ${_formatDate(cultivo.fechaCosecha!)}',
                    style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                  ),
                ],
              ),
            ],

            if (cultivo.fechaSiembra.isNotEmpty) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.agriculture, size: 14, color: Colors.grey[600]),
                  const SizedBox(width: 4),
                  Text(
                    'Siembra: ${_formatDate(cultivo.fechaSiembra)}',
                    style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                  ),
                ],
              ),
              const SizedBox(height: 4),
            ],

            if (cultivo.notas?.isNotEmpty == true) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.note, size: 16, color: Colors.grey[600]),
                  const SizedBox(width: 4),
                  Expanded(
                    child: Text(
                      'Notas: ${cultivo.notas}',
                      style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                    ),
                  ),
                ],
              ),
            ],

            const SizedBox(height: 12),

            // Información de riesgo
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: color.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.info_outline, size: 16, color: color),
                      const SizedBox(width: 4),
                      Text(
                        'Información de Riesgo',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: color,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (cultivo.fechaSiembra.isNotEmpty) ...[
                    Row(
                      children: [
                        Icon(
                          Icons.agriculture,
                          size: 14,
                          color: Colors.grey[600],
                        ),
                        const SizedBox(width: 4),
                        Text(
                          'Siembra: ${_formatDate(cultivo.fechaSiembra)}',
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                  ],
                  Row(
                    children: [
                      Icon(Icons.warning_amber, size: 14, color: color),
                      const SizedBox(width: 4),
                      Text(
                        'Estado: Riesgo activado',
                        style: TextStyle(
                          fontSize: 11,
                          color: color,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                  if (cultivo.fechaCosecha != null) ...[
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        Icon(Icons.schedule, size: 14, color: Colors.grey[600]),
                        const SizedBox(width: 4),
                        Text(
                          'Tiempo para cosecha: ${_calcularDiasParaCosecha(cultivo.fechaCosecha!)} días',
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
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
            child: Icon(Icons.check_circle, size: 64, color: Colors.green[700]),
          ),
          const SizedBox(height: 24),
          Text(
            '¡No hay cultivos en riesgo!',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.grey[800],
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Todos los cultivos están en buen estado',
            style: TextStyle(fontSize: 14, color: Colors.grey[600]),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: _loadCultivosEnRiesgo,
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
      return '${date.day.toString().padLeft(2, '0')}/${date.month.toString().padLeft(2, '0')}/${date.year}';
    } catch (e) {
      return dateString;
    }
  }

  String _calcularDiasParaCosecha(String fechaCosechaStr) {
    try {
      final fechaCosecha = DateTime.parse(fechaCosechaStr);
      final dias = fechaCosecha.difference(DateTime.now()).inDays;
      return dias.toString();
    } catch (e) {
      return 'N/A';
    }
  }
}
