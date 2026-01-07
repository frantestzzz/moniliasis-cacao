// lib/View/historial_riesgos_page.dart
import 'package:flutter/material.dart';
import 'package:amgeca/classes/cultivo.dart';

class HistorialRiesgosPage extends StatefulWidget {
  final List<Cultivo> cultivos;

  const HistorialRiesgosPage({super.key, required this.cultivos});

  @override
  State<HistorialRiesgosPage> createState() => _HistorialRiesgosPageState();
}

class _HistorialRiesgosPageState extends State<HistorialRiesgosPage> {
  Cultivo? _cultivoExpandido;

  @override
  Widget build(BuildContext context) {
    // Separar cultivos en dos grupos
    final cultivosEnRiesgoActual = widget.cultivos
        .where((c) => c.esEnRiesgo && c.razonRiesgo != null)
        .toList();

    final cultivosConHistorial =
        widget.cultivos.where((c) => c.tieneHistorialRiesgos).toList()
          ..sort((a, b) {
            // Ordenar por fecha del último riesgo (más reciente primero)
            final historialA = a.getHistorialRiesgos();
            final historialB = b.getHistorialRiesgos();
            if (historialA.isEmpty) return 1;
            if (historialB.isEmpty) return -1;

            final fechaFinA = historialA.last['fechaFin'] as String? ?? '';
            final fechaFinB = historialB.last['fechaFin'] as String? ?? '';
            return fechaFinB.compareTo(fechaFinA);
          });

    // Contar total de cultivos únicos que han estado en riesgo
    final cultivosUnicos = <int>{};
    for (var cultivo in widget.cultivos) {
      if (cultivo.esEnRiesgo || cultivo.tieneHistorialRiesgos) {
        cultivosUnicos.add(cultivo.id!);
      }
    }

    final totalCultivosConRiesgos = cultivosUnicos.length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Historial Completo de Riesgos'),
        backgroundColor: Colors.orange[700],
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: totalCultivosConRiesgos == 0
          ? _buildEmptyState()
          : SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Estadística general
                  _buildEstadisticaGeneral(totalCultivosConRiesgos),

                  // Sección: Cultivos actualmente en riesgo
                  if (cultivosEnRiesgoActual.isNotEmpty) ...[
                    _buildSectionHeader(
                      '🔴 ACTUALMENTE EN RIESGO',
                      cultivosEnRiesgoActual.length,
                      Colors.red,
                    ),
                    ...cultivosEnRiesgoActual.map(
                      (cultivo) => _buildCultivoResumidoCard(cultivo, true),
                    ),
                  ],

                  // Sección: Historial de riesgos pasados
                  if (cultivosConHistorial.isNotEmpty) ...[
                    _buildSectionHeader(
                      '📋 HISTORIAL DE RIESGOS',
                      cultivosConHistorial.length,
                      Colors.blue,
                    ),
                    ...cultivosConHistorial.map(
                      (cultivo) => _buildCultivoResumidoCard(cultivo, false),
                    ),
                  ],

                  const SizedBox(height: 16),
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
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.orange[50],
                shape: BoxShape.circle,
                border: Border.all(color: Colors.orange[200]!, width: 3),
              ),
              child: Icon(Icons.history, size: 64, color: Colors.orange[400]),
            ),
            const SizedBox(height: 24),
            Text(
              'Sin riesgos registrados',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.grey[700],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Ningún cultivo ha estado en riesgo.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEstadisticaGeneral(int total) {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.orange[600]!, Colors.orange[400]!],
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.orange.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(
              Icons.warning_amber_rounded,
              size: 32,
              color: Colors.white,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Total de cultivos con riesgos',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.white70,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '$total cultivo${total != 1 ? 's' : ''}',
                  style: const TextStyle(
                    fontSize: 24,
                    color: Colors.white,
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

  Widget _buildSectionHeader(String titulo, int cantidad, Color color) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3), width: 2),
      ),
      child: Row(
        children: [
          Icon(Icons.label, color: color, size: 20),
          const SizedBox(width: 8),
          Text(
            titulo,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              '$cantidad',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCultivoEnRiesgoCard(Cultivo cultivo) {
    final historial = cultivo.getHistorialRiesgos();
    final vecesEnRiesgo = historial.length + 1; // +1 por el riesgo actual
    final numeroVez = _getNumeroVez(vecesEnRiesgo);

    Color gravedadColor = Colors.orange;
    IconData gravedadIcon = Icons.warning;

    switch (cultivo.nivelGravedad?.toLowerCase()) {
      case 'leve':
        gravedadColor = Colors.green;
        gravedadIcon = Icons.sentiment_satisfied;
        break;
      case 'moderado':
        gravedadColor = Colors.orange;
        gravedadIcon = Icons.sentiment_neutral;
        break;
      case 'grave':
        gravedadColor = Colors.red;
        gravedadIcon = Icons.sentiment_very_dissatisfied;
        break;
    }

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.red.withOpacity(0.5), width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header del cultivo
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.red[50],
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(16),
                topRight: Radius.circular(16),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.agriculture, color: Colors.red[700], size: 24),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            cultivo.nombre,
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.black87,
                            ),
                          ),
                          Text(
                            'EN RIESGO - $numeroVez',
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.red[700],
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.red,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Text(
                        'ACTIVO',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Riesgo actual
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: gravedadColor.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: gravedadColor.withOpacity(0.3),
                        ),
                      ),
                      child: Icon(gravedadIcon, color: gravedadColor, size: 20),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            '⚠️ Riesgo Actual',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            cultivo.tipoRiesgo ?? 'Tipo no especificado',
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Colors.black87,
                            ),
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
                        color: gravedadColor.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: gravedadColor.withOpacity(0.3),
                        ),
                      ),
                      child: Text(
                        (cultivo.nivelGravedad ?? 'N/A').toUpperCase(),
                        style: TextStyle(
                          color: gravedadColor,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey[50],
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey[200]!),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Razón:',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.grey[600],
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        cultivo.razonRiesgo ?? 'Sin razón especificada',
                        style: const TextStyle(
                          fontSize: 13,
                          color: Colors.black87,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildFechaChip(
                        'Desde',
                        cultivo.fechaInicioRiesgo ?? 'N/A',
                        Icons.calendar_today,
                        Colors.red,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: _buildFechaChip(
                        'Estado',
                        'En curso',
                        Icons.update,
                        Colors.orange,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Historial previo (si existe)
          if (historial.isNotEmpty) ...[
            const Divider(height: 1),
            Container(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.history, color: Colors.grey[700], size: 18),
                      const SizedBox(width: 8),
                      Text(
                        'Historial anterior (${historial.length})',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[700],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  ...historial.asMap().entries.map((entry) {
                    final index = entry.key;
                    final riesgo = entry.value;
                    return _buildRiesgoHistorialItem(riesgo, index + 1);
                  }),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildCultivoHistorialCard(Cultivo cultivo) {
    final historial = cultivo.getHistorialRiesgos();
    final vecesEnRiesgo =
        historial.length + 1; // +1 para incluir el conteo total correcto
    final numeroVez = _getNumeroVez(vecesEnRiesgo);

    // Determinar estado y color
    String estadoLabel;
    Color estadoColor;
    if (cultivo.esCosechado) {
      estadoLabel = 'COSECHADO';
      estadoColor = Colors.amber;
    } else if (cultivo.esInactivo) {
      estadoLabel = 'INACTIVO';
      estadoColor = Colors.grey;
    } else {
      estadoLabel = 'ACTIVO';
      estadoColor = Colors.green;
    }

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header del cultivo
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.blue[50],
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(16),
                topRight: Radius.circular(16),
              ),
            ),
            child: Row(
              children: [
                Icon(Icons.agriculture, color: Colors.blue[700], size: 24),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        cultivo.nombre,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                      Text(
                        '$estadoLabel - $numeroVez',
                        style: TextStyle(
                          fontSize: 13,
                          color: estadoColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: estadoColor,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    estadoLabel,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Lista de riesgos
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.list, color: Colors.grey[700], size: 18),
                    const SizedBox(width: 8),
                    Text(
                      'Riesgos registrados (${historial.length})',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.bold,
                        color: Colors.grey[700],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                ...historial.asMap().entries.map((entry) {
                  final index = entry.key;
                  final riesgo = entry.value;
                  return _buildRiesgoHistorialItem(riesgo, index + 1);
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiesgoHistorialItem(Map<String, dynamic> riesgo, int numero) {
    final tipo = riesgo['tipo'] as String? ?? 'Desconocido';
    final razon = riesgo['razon'] as String? ?? 'Sin razón';
    final fechaInicio = riesgo['fechaInicio'] as String? ?? 'N/A';
    final fechaFin = riesgo['fechaFin'] as String? ?? 'N/A';
    final nivelGravedad = riesgo['nivelGravedad'] as String? ?? '';

    Color gravedadColor = Colors.grey;
    switch (nivelGravedad.toLowerCase()) {
      case 'leve':
        gravedadColor = Colors.green;
        break;
      case 'moderado':
        gravedadColor = Colors.orange;
        break;
      case 'grave':
        gravedadColor = Colors.red;
        break;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: gravedadColor.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: gravedadColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '#$numero',
                  style: TextStyle(
                    color: gravedadColor,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  tipo,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
              ),
              if (nivelGravedad.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: gravedadColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    nivelGravedad.toUpperCase(),
                    style: TextStyle(
                      color: gravedadColor,
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            razon,
            style: TextStyle(fontSize: 12, color: Colors.grey[700]),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _buildFechaChipSmall(
                  'Inicio',
                  fechaInicio,
                  Icons.calendar_today,
                  Colors.red,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _buildFechaChipSmall(
                  'Fin',
                  fechaFin,
                  Icons.event_available,
                  Colors.green,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildFechaChip(
    String label,
    String fecha,
    IconData icon,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(height: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              color: color,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            fecha,
            style: TextStyle(
              fontSize: 11,
              color: color,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildFechaChipSmall(
    String label,
    String fecha,
    IconData icon,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 12),
          const SizedBox(width: 4),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 9,
                    color: color,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  fecha,
                  style: TextStyle(
                    fontSize: 10,
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _getNumeroVez(int veces) {
    if (veces == 1) return '1ra vez en riesgo';
    if (veces == 2) return '2da vez en riesgo';
    if (veces == 3) return '3ra vez en riesgo';
    return '${veces}ta vez en riesgo';
  }

  Widget _buildCultivoResumidoCard(Cultivo cultivo, bool estaEnRiesgo) {
    final historial = cultivo.getHistorialRiesgos();
    final vecesEnRiesgo = cultivo.vecesEnRiesgo;
    final numeroVez = _getNumeroVez(vecesEnRiesgo);
    final estaExpandido = _cultivoExpandido?.id == cultivo.id;

    // Determinar estado y color
    String estadoLabel;
    Color estadoColor;
    if (estaEnRiesgo) {
      estadoLabel = 'EN RIESGO';
      estadoColor = Colors.red;
    } else if (cultivo.esCosechado) {
      estadoLabel = 'COSECHADO';
      estadoColor = Colors.amber;
    } else if (cultivo.esInactivo) {
      estadoLabel = 'INACTIVO';
      estadoColor = Colors.grey;
    } else {
      estadoLabel = 'ACTIVO';
      estadoColor = Colors.green;
    }

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: estaExpandido ? 6 : 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: estaEnRiesgo
              ? Colors.red.withOpacity(0.5)
              : Colors.transparent,
          width: estaEnRiesgo ? 2 : 1,
        ),
      ),
      child: InkWell(
        onTap: () {
          setState(() {
            _cultivoExpandido = estaExpandido ? null : cultivo;
          });
        },
        borderRadius: BorderRadius.circular(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header resumido
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: estaEnRiesgo ? Colors.red[50] : Colors.blue[50],
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  topRight: Radius.circular(16),
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.agriculture,
                    color: estaEnRiesgo ? Colors.red[700] : Colors.blue[700],
                    size: 24,
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
                            color: Colors.black87,
                          ),
                        ),
                        Text(
                          '$estadoLabel - $numeroVez',
                          style: TextStyle(
                            fontSize: 13,
                            color: estadoColor,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: estadoColor,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      estadoLabel,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Icon(
                    estaExpandido ? Icons.expand_less : Icons.expand_more,
                    color: Colors.grey[600],
                  ),
                ],
              ),
            ),

            // Información adicional (solo si está expandido)
            if (estaExpandido) ...[
              Container(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Riesgo actual (si aplica)
                    if (estaEnRiesgo) ...[
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.orange[50],
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.orange[200]!),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Icon(
                                  Icons.warning,
                                  color: Colors.orange[700],
                                  size: 20,
                                ),
                                const SizedBox(width: 8),
                                const Text(
                                  '⚠️ Riesgo Actual',
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.black87,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Tipo: ${cultivo.tipoRiesgo ?? 'No especificado'}',
                              style: const TextStyle(fontSize: 13),
                            ),
                            Text(
                              'Razón: ${cultivo.razonRiesgo ?? 'No especificada'}',
                              style: const TextStyle(fontSize: 13),
                            ),
                            Text(
                              'Desde: ${cultivo.fechaInicioRiesgo ?? 'No especificada'}',
                              style: const TextStyle(fontSize: 13),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],

                    // Historial completo
                    if (historial.isNotEmpty) ...[
                      Row(
                        children: [
                          Icon(
                            Icons.history,
                            color: Colors.grey[700],
                            size: 18,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'Historial completo (${historial.length} eventos)',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.bold,
                              color: Colors.grey[700],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      ...historial.asMap().entries.map((entry) {
                        final index = entry.key;
                        final riesgo = entry.value;
                        return _buildRiesgoDetallado(riesgo, index + 1);
                      }),
                    ],
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildRiesgoDetallado(Map<String, dynamic> riesgo, int numero) {
    final tipo = riesgo['tipo'] as String? ?? 'Desconocido';
    final razon = riesgo['razon'] as String? ?? 'Sin razón';
    final fechaInicio = riesgo['fechaInicio'] as String? ?? 'N/A';
    final fechaFin = riesgo['fechaFin'] as String? ?? 'N/A';
    final nivelGravedad = riesgo['nivelGravedad'] as String? ?? '';

    Color gravedadColor = Colors.grey;
    switch (nivelGravedad.toLowerCase()) {
      case 'leve':
        gravedadColor = Colors.green;
        break;
      case 'moderado':
        gravedadColor = Colors.orange;
        break;
      case 'grave':
        gravedadColor = Colors.red;
        break;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: gravedadColor.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: gravedadColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '#$numero',
                  style: TextStyle(
                    color: gravedadColor,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  tipo,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
              ),
              if (nivelGravedad.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: gravedadColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    nivelGravedad.toUpperCase(),
                    style: TextStyle(
                      color: gravedadColor,
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Razón: $razon',
            style: const TextStyle(fontSize: 12, color: Colors.black87),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Icon(Icons.calendar_today, color: Colors.red, size: 14),
              const SizedBox(width: 4),
              Text(
                'Inicio: $fechaInicio',
                style: const TextStyle(fontSize: 11, color: Colors.red),
              ),
              const SizedBox(width: 16),
              Icon(Icons.event_available, color: Colors.green, size: 14),
              const SizedBox(width: 4),
              Text(
                'Fin: $fechaFin',
                style: const TextStyle(fontSize: 11, color: Colors.green),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
