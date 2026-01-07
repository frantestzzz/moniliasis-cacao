// lib/providers/stats_provider.dart
import 'package:flutter/material.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class StatsProvider extends ChangeNotifier {
  final BasedatoHelper _dbHelper = BasedatoHelper.instance;

  double _totalIngresos = 0.0;
  double _totalEgresos = 0.0;
  double _totalVentas = 0.0;
  int _ventasMesActual = 0;
  int _ventasMesAnterior = 0;
  bool _isLoading = false;

  double get totalIngresos => _totalIngresos;
  double get totalEgresos => _totalEgresos;
  double get totalVentas => _totalVentas;
  double get rentabilidad => _totalIngresos - _totalEgresos;
  bool get isLoading => _isLoading;

  int get monthlyVariation {
    if (_ventasMesAnterior == 0) return 0;
    return (((_ventasMesActual - _ventasMesAnterior) / _ventasMesAnterior) *
            100)
        .round();
  }

  String get monthlyTrend => monthlyVariation >= 0 ? '↑' : '↓';

  Future<void> loadStats() async {
    _isLoading = true;
    notifyListeners();

    try {
      // Calcular ingresos totales desde cultivos
      final cultivos = await _dbHelper.getAllCultivos();
      _totalIngresos = cultivos.fold(
        0.0,
        (sum, c) => sum + ((c['ingresos'] as num?)?.toDouble() ?? 0.0),
      );
      _totalEgresos = cultivos.fold(
        0.0,
        (sum, c) => sum + ((c['egresos'] as num?)?.toDouble() ?? 0.0),
      );

      // Calcular ventas totales
      _totalVentas = await _dbHelper.getTotalVentas();

      // Calcular ventas del mes actual vs mes anterior
      final ahora = DateTime.now();
      final inicioMesActual = DateTime(
        ahora.year,
        ahora.month,
        1,
      ).toIso8601String();
      final finMesActual = DateTime(
        ahora.year,
        ahora.month + 1,
        0,
      ).toIso8601String();

      final inicioMesAnterior = DateTime(
        ahora.year,
        ahora.month - 1,
        1,
      ).toIso8601String();
      final finMesAnterior = DateTime(
        ahora.year,
        ahora.month,
        0,
      ).toIso8601String();

      final ventasMesActual = await _dbHelper.getVentasPorFechas(
        inicioMesActual,
        finMesActual,
      );
      final ventasMesAnterior = await _dbHelper.getVentasPorFechas(
        inicioMesAnterior,
        finMesAnterior,
      );

      _ventasMesActual = ventasMesActual.length;
      _ventasMesAnterior = ventasMesAnterior.length;
    } catch (e) {
      debugPrint('Error al cargar estadísticas: $e');
    }

    _isLoading = false;
    notifyListeners();
  }

  double getEfficiency({required int current, required int previous}) {
    if (previous == 0) return 100;
    return ((current - previous) / previous) * 100;
  }

  // Calcular rentabilidad por cultivo
  Future<Map<String, double>> getRentabilidadPorCultivo() async {
    final cultivos = await _dbHelper.getAllCultivos();
    final Map<String, double> rentabilidad = {};

    for (final cultivo in cultivos) {
      final nombre = cultivo['nombre'] as String;
      final ingresos = (cultivo['ingresos'] as num?)?.toDouble() ?? 0.0;
      final egresos = (cultivo['egresos'] as num?)?.toDouble() ?? 0.0;
      rentabilidad[nombre] = ingresos - egresos;
    }

    return rentabilidad;
  }
}
