import 'package:flutter/material.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:amgeca/classes/cultivo.dart';

class CropsProvider extends ChangeNotifier {
  final BasedatoHelper _dbHelper = BasedatoHelper.instance;
  final List<Cultivo> _cultivos = [];
  bool _isLoading = false;

  List<Cultivo> get cultivos => List.unmodifiable(_cultivos);
  bool get isLoading => _isLoading;
  int get activeCount => _cultivos.where((c) => c.esActivo).length;
  int get enRiesgoCount => _cultivos.where((c) => c.esEnRiesgo).length;
  int get harvestedCount => _cultivos.where((c) => c.esCosechado).length;
  int get inactiveCount => _cultivos.where((c) => c.esInactivo).length;
  double get totalArea => _cultivos.fold(0.0, (prev, c) => prev + c.area);

  Future<void> loadCultivos() async {
    _isLoading = true;
    notifyListeners();

    final rows = await _dbHelper.getAllCultivos();
    _cultivos
      ..clear()
      ..addAll(rows.map((row) => Cultivo.fromMap(row)));

    _isLoading = false;
    notifyListeners();
  }

  List<Cultivo> displayForEstado(String estado) {
    return _cultivos.where((c) => c.estado == estado).toList();
  }

  List<Cultivo> upcomingHarvests({int limit = 3}) {
    final today = DateTime.now();
    final list =
        _cultivos.where((c) {
          if (c.fechaCosecha == null) return false;
          final parsed = DateTime.tryParse(c.fechaCosecha!);
          return parsed != null && parsed.isAfter(today);
        }).toList()..sort((a, b) {
          final aDate = DateTime.tryParse(a.fechaCosecha!) ?? today;
          final bDate = DateTime.tryParse(b.fechaCosecha!) ?? today;
          return aDate.compareTo(bDate);
        });
    return list.take(limit).toList();
  }
}
