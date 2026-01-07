// lib/providers/alerts_provider.dart
import 'package:flutter/material.dart';
import 'package:amgeca/Model/alert_item.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class AlertsProvider extends ChangeNotifier {
  final BasedatoHelper _dbHelper = BasedatoHelper.instance;
  final List<AlertItem> _alerts = [];
  bool _isLoading = false;

  List<AlertItem> get alerts => List.unmodifiable(_alerts);
  bool get isLoading => _isLoading;

  int get totalCritical =>
      _alerts.where((a) => a.severity == 'critical' && !a.resuelta).length;
  int get totalWarning =>
      _alerts.where((a) => a.severity == 'warning' && !a.resuelta).length;
  int get totalInfo =>
      _alerts.where((a) => a.severity == 'info' && !a.resuelta).length;
  int get newRiskAlerts => totalCritical + totalWarning;

  // Método principal - compatibilidad con nombre antiguo
  Future<void> loadMockAlerts() async {
    await loadAlerts();
  }

  Future<void> loadAlerts() async {
    _isLoading = true;
    notifyListeners();

    try {
      // Generar alertas automáticas desde cultivos
      await _dbHelper.generarAlertasAutomaticas();

      // Cargar alertas de la BD
      final rows = await _dbHelper.getAllAlertas();

      _alerts.clear();
      for (final row in rows) {
        _alerts.add(
          AlertItem(
            id: row['id'].toString(),
            title: row['titulo'] as String,
            message: row['mensaje'] as String,
            severity: row['severidad'] as String,
            cultivoName: await _getCultivoName(row['cultivoId'] as int?),
            date: DateTime.parse(row['fecha'] as String),
            targetRoute: row['rutaDestino'] as String?,
            resuelta: (row['resuelta'] as int) == 1,
          ),
        );
      }
    } catch (e) {
      debugPrint('Error al cargar alertas: $e');
    }

    _isLoading = false;
    notifyListeners();
  }

  Future<String> _getCultivoName(int? cultivoId) async {
    if (cultivoId == null) return 'Sistema';

    try {
      final cultivos = await _dbHelper.getAllCultivos();
      final cultivo = cultivos.firstWhere(
        (c) => c['id'] == cultivoId,
        orElse: () => {'nombre': 'Cultivo desconocido'},
      );
      return cultivo['nombre'] as String;
    } catch (e) {
      return 'Cultivo desconocido';
    }
  }

  Future<void> marcarResuelta(String alertId) async {
    try {
      await _dbHelper.marcarAlertaResuelta(int.parse(alertId));
      await loadAlerts();
    } catch (e) {
      debugPrint('Error al marcar alerta resuelta: $e');
    }
  }

  Future<void> eliminarAlerta(String alertId) async {
    try {
      await _dbHelper.eliminarAlerta(int.parse(alertId));
      await loadAlerts();
    } catch (e) {
      debugPrint('Error al eliminar alerta: $e');
    }
  }

  List<AlertItem> recentNotifications({int limit = 5}) {
    final copy = _alerts.where((a) => !a.resuelta).toList();
    copy.sort((a, b) => b.date.compareTo(a.date));
    return copy.take(limit).toList();
  }

  List<AlertItem> getAlertasPorSeveridad(String severidad) {
    return _alerts
        .where((a) => a.severity == severidad && !a.resuelta)
        .toList();
  }
}
