// lib/services/weather_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'dart:developer' as developer;

class WeatherData {
  final double temperatura;
  final double humedadRelativa;
  final double velocidadViento;
  final int codigoClima;
  final String descripcionClima;
  final List<PronosticoHora> pronosticoHorario;
  final double presion;
  final double puntoRocio;
  final double visibilidad;
  final int indiceUV;

  WeatherData({
    required this.temperatura,
    required this.humedadRelativa,
    required this.velocidadViento,
    required this.codigoClima,
    required this.descripcionClima,
    required this.pronosticoHorario,
    required this.presion,
    required this.puntoRocio,
    required this.visibilidad,
    required this.indiceUV,
  });

  String get iconoClima {
    if (codigoClima == 0) return '☀️';
    if (codigoClima <= 3) return '⛅';
    if (codigoClima <= 48) return '🌫️';
    if (codigoClima <= 67) return '🌧️';
    if (codigoClima <= 77) return '🌨️';
    if (codigoClima <= 82) return '⛈️';
    if (codigoClima <= 99) return '⚡';
    return '🌤️';
  }

  String get calidadAire {
    if (indiceUV <= 2) return 'Bajo';
    if (indiceUV <= 5) return 'Moderado';
    if (indiceUV <= 7) return 'Alto';
    if (indiceUV <= 10) return 'Muy alto';
    return 'Extremo';
  }

  String get descripcionViento {
    if (velocidadViento < 5) return 'Brisa ligera';
    if (velocidadViento < 15) return 'Ligera brisa';
    if (velocidadViento < 25) return 'Brisa moderada';
    if (velocidadViento < 35) return 'Viento fuerte';
    return 'Viento muy fuerte';
  }

  String get descripcionPuntoRocio {
    if (puntoRocio < 10) return 'Seco';
    if (puntoRocio < 15) return 'Confortable';
    if (puntoRocio < 18) return 'Se siente bien';
    if (puntoRocio < 21) return 'Sofocante';
    return 'Muy húmedo';
  }

  bool get esAptoParaPulverizar {
    bool tempAdecuada = temperatura >= 18 && temperatura <= 28;
    bool humedadAdecuada = humedadRelativa >= 50 && humedadRelativa <= 80;
    bool vientoAdecuado = velocidadViento < 15;
    bool sinLluvia = codigoClima < 50;

    return tempAdecuada && humedadAdecuada && vientoAdecuado && sinLluvia;
  }

  String get recomendacionPulverizacion {
    if (esAptoParaPulverizar) {
      return '✅ Condiciones óptimas para pulverizar';
    }

    List<String> problemas = [];
    if (temperatura < 18) problemas.add('Temperatura muy baja');
    if (temperatura > 28) problemas.add('Temperatura muy alta');
    if (humedadRelativa < 50) problemas.add('Humedad muy baja');
    if (humedadRelativa > 80) problemas.add('Humedad muy alta');
    if (velocidadViento >= 15) problemas.add('Viento muy fuerte');
    if (codigoClima >= 50) problemas.add('Condiciones de lluvia');

    return '⚠️ No recomendado: ${problemas.join(', ')}';
  }
}

class PronosticoHora {
  final DateTime hora;
  final double temperatura;
  final double probabilidadLluvia;
  final int codigoClima;

  PronosticoHora({
    required this.hora,
    required this.temperatura,
    required this.probabilidadLluvia,
    required this.codigoClima,
  });

  String get iconoClima {
    if (codigoClima == 0) return '☀️';
    if (codigoClima <= 3) return '⛅';
    if (codigoClima <= 48) return '🌫️';
    if (codigoClima <= 67) return '🌧️';
    if (codigoClima <= 77) return '🌨️';
    if (codigoClima <= 82) return '⛈️';
    if (codigoClima <= 99) return '⚡';
    return '🌤️';
  }
}

class WeatherService {
  static final WeatherService _instance = WeatherService._internal();
  factory WeatherService() => _instance;
  WeatherService._internal();

  Future<WeatherData?> obtenerClima(double lat, double lon) async {
    try {
      final url = Uri.parse(
        'https://api.open-meteo.com/v1/forecast?'
        'latitude=$lat&longitude=$lon'
        '&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,surface_pressure,dew_point_2m,visibility,uv_index'
        '&hourly=temperature_2m,precipitation_probability,weather_code'
        '&timezone=auto'
        '&forecast_days=1',
      );

      developer.log('🌐 Consultando clima: $url', name: 'WeatherService');

      final response = await http.get(url);

      if (response.statusCode != 200) {
        developer.log(
          '❌ Error HTTP: ${response.statusCode}',
          name: 'WeatherService',
        );
        return null;
      }

      final data = json.decode(response.body);

      // Datos actuales
      final current = data['current'];
      final hourly = data['hourly'];

      // Pronóstico por hora (próximas 12 horas)
      List<PronosticoHora> pronostico = [];
      for (int i = 0; i < 12 && i < hourly['time'].length; i++) {
        pronostico.add(
          PronosticoHora(
            hora: DateTime.parse(hourly['time'][i]),
            temperatura: (hourly['temperature_2m'][i] ?? 0).toDouble(),
            probabilidadLluvia: (hourly['precipitation_probability'][i] ?? 0)
                .toDouble(),
            codigoClima: hourly['weather_code'][i] ?? 0,
          ),
        );
      }

      final weatherData = WeatherData(
        temperatura: (current['temperature_2m'] ?? 0).toDouble(),
        humedadRelativa: (current['relative_humidity_2m'] ?? 0).toDouble(),
        velocidadViento: (current['wind_speed_10m'] ?? 0).toDouble(),
        codigoClima: current['weather_code'] ?? 0,
        descripcionClima: _obtenerDescripcionClima(
          current['weather_code'] ?? 0,
        ),
        pronosticoHorario: pronostico,
        presion: (current['surface_pressure'] ?? 1013.25).toDouble(),
        puntoRocio: (current['dew_point_2m'] ?? 0).toDouble(),
        visibilidad: (current['visibility'] ?? 10000).toDouble() / 1000,
        indiceUV: (current['uv_index'] ?? 0).toInt(),
      );

      developer.log(
        '✅ Clima obtenido: ${weatherData.temperatura}°C',
        name: 'WeatherService',
      );
      return weatherData;
    } catch (e) {
      developer.log('❌ Error al obtener clima: $e', name: 'WeatherService');
      return null;
    }
  }

  String _obtenerDescripcionClima(int codigo) {
    switch (codigo) {
      case 0:
        return 'Despejado';
      case 1:
        return 'Principalmente despejado';
      case 2:
        return 'Parcialmente nublado';
      case 3:
        return 'Nublado';
      case 45:
      case 48:
        return 'Neblina';
      case 51:
      case 53:
      case 55:
        return 'Llovizna';
      case 61:
      case 63:
      case 65:
        return 'Lluvia';
      case 71:
      case 73:
      case 75:
        return 'Nevada';
      case 77:
        return 'Granizo';
      case 80:
      case 81:
      case 82:
        return 'Aguacero';
      case 85:
      case 86:
        return 'Nevada intensa';
      case 95:
        return 'Tormenta';
      case 96:
      case 99:
        return 'Tormenta con granizo';
      default:
        return 'Desconocido';
    }
  }
}
