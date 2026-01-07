import 'package:flutter/material.dart';
import 'package:amgeca/services/weather_service.dart';

class WeatherProvider extends ChangeNotifier {
  final WeatherService _service = WeatherService();
  WeatherData? _current;
  bool _isLoading = false;
  String? _error;

  WeatherData? get current => _current;
  bool get isLoading => _isLoading;
  String? get error => _error;

  // Fallback mock data
  double get currentTemperature => _current?.temperatura ?? 22.5;
  double get humidity => _current?.humedadRelativa ?? 60;
  double get rainChance => _current?.pronosticoHorario.isNotEmpty == true
      ? _current!.pronosticoHorario.first.probabilidadLluvia
      : 15;

  Future<void> refresh({double lat = -12.0464, double lon = -77.0428}) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      final data = await _service.obtenerClima(lat, lon);
      if (data != null) {
        _current = data;
      } else {
        _error = 'No se pudo obtener el clima';
      }
    } catch (e) {
      _error = 'Error: $e';
    }

    _isLoading = false;
    notifyListeners();
  }

  String get recommendation =>
      _current?.recomendacionPulverizacion ?? 'Cargando...';
  bool get isGoodForSpraying => _current?.esAptoParaPulverizar ?? false;
}

class Temperature {
  final double current;
  final double humidity;
  final double rainChance;

  Temperature({
    required this.current,
    required this.humidity,
    required this.rainChance,
  });
}
