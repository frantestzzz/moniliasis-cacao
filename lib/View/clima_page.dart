// lib/View/clima_page.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../services/location_service.dart';
import '../services/weather_service.dart';
import '../services/notification_service.dart';
import '../services/deepseek_service.dart';

class ClimaPage extends StatefulWidget {
  const ClimaPage({Key? key}) : super(key: key);

  @override
  State<ClimaPage> createState() => _ClimaPageState();
}

class _ClimaPageState extends State<ClimaPage> {
  final LocationService _locationService = LocationService();
  final WeatherService _weatherService = WeatherService();
  final NotificationService _notificationService = NotificationService();
  final DeepSeekService _deepSeekService = DeepSeekService();

  WeatherData? _weatherData;
  String? _ubicacion;
  bool _cargando = false;
  String? _error;
  String? _recomendacionIA;
  bool _cargandoIA = false;

  @override
  void initState() {
    super.initState();
    _notificationService.initialize();
    _cargarDatosClima();
  }

  Future<void> _cargarDatosClima() async {
    setState(() {
      _cargando = true;
      _error = null;
    });

    try {
      final position = await _locationService.obtenerUbicacionActual();

      if (position == null) {
        if (mounted) {
          setState(() {
            _error = 'No se pudo obtener la ubicación. Verifica los permisos.';
            _cargando = false;
          });
        }
        return;
      }

      final weatherData = await _weatherService.obtenerClima(
        position.latitude,
        position.longitude,
      );

      if (weatherData == null) {
        if (mounted) {
          setState(() {
            _error = 'No se pudo obtener la información del clima.';
            _cargando = false;
          });
        }
        return;
      }

      final ubicacion = await _locationService.obtenerNombreCiudad(
        position.latitude,
        position.longitude,
      );

      if (mounted) {
        setState(() {
          _weatherData = weatherData;
          _ubicacion = ubicacion;
          _cargando = false;
        });
      }

      if (!weatherData.esAptoParaPulverizar) {
        await _notificationService.mostrarNotificacionClimaDesfavorable(
          titulo: '⚠️ Condiciones Desfavorables',
          mensaje: weatherData.recomendacionPulverizacion,
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Error: $e';
          _cargando = false;
        });
      }
    }
  }

  Future<void> _obtenerRecomendacionIA() async {
    if (_weatherData == null) return;

    setState(() => _cargandoIA = true);

    final recomendacion = await _deepSeekService.obtenerRecomendacionesClima(
      temperatura: _weatherData!.temperatura,
      humedad: _weatherData!.humedadRelativa,
      viento: _weatherData!.velocidadViento,
      descripcionClima: _weatherData!.descripcionClima,
      esAptoParaPulverizar: _weatherData!.esAptoParaPulverizar,
    );

    if (mounted) {
      setState(() {
        _recomendacionIA = recomendacion;
        _cargandoIA = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Clima'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _cargarDatosClima,
            tooltip: 'Actualizar',
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _cargarDatosClima,
        child: _cargando
            ? const Center(child: CircularProgressIndicator(color: Colors.blue))
            : _error != null
            ? _buildError()
            : _weatherData == null
            ? _buildVacio()
            : _buildContenido(),
      ),
    );
  }

  Widget _buildError() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _error!,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _cargarDatosClima,
              icon: const Icon(Icons.refresh),
              label: const Text('Reintentar'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVacio() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.cloud_off, size: 64, color: Colors.grey[400]),
          const SizedBox(height: 16),
          Text(
            'No hay datos del clima',
            style: TextStyle(fontSize: 16, color: Colors.grey[600]),
          ),
        ],
      ),
    );
  }

  Widget _buildContenido() {
    return SingleChildScrollView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Ubicación
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.location_on, color: Colors.blue, size: 20),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  _ubicacion ?? 'Ubicación desconocida',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Clima actual
          _buildClimaActual(),
          const SizedBox(height: 24),

          // Condiciones para pulverización
          _buildCondicionesPulverizacion(),
          const SizedBox(height: 24),

          // Pronóstico por hora
          _buildPronosticoHorario(),
          const SizedBox(height: 24),

          // ✨ NUEVA SECCIÓN: Detalles Meteorológicos
          _buildDetallesMeteorologicos(),
          const SizedBox(height: 24),

          // Botón de recomendaciones IA
          _buildBotonIA(),

          // Recomendaciones IA
          if (_recomendacionIA != null) ...[
            const SizedBox(height: 16),
            _buildRecomendacionesIA(),
          ],
        ],
      ),
    );
  }

  Widget _buildClimaActual() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Colors.blue[400]!, Colors.blue[700]!],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.blue.withOpacity(0.3),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Text(_weatherData!.iconoClima, style: const TextStyle(fontSize: 80)),
          const SizedBox(height: 8),
          Text(
            '${_weatherData!.temperatura.toStringAsFixed(1)}°C',
            style: const TextStyle(
              fontSize: 48,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          Text(
            _weatherData!.descripcionClima,
            style: const TextStyle(fontSize: 20, color: Colors.white),
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildIndicador(
                Icons.water_drop,
                '${_weatherData!.humedadRelativa.toStringAsFixed(0)}%',
                'Humedad',
              ),
              _buildIndicador(
                Icons.air,
                '${_weatherData!.velocidadViento.toStringAsFixed(1)} km/h',
                'Viento',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildIndicador(IconData icono, String valor, String etiqueta) {
    return Column(
      children: [
        Icon(icono, color: Colors.white, size: 32),
        const SizedBox(height: 4),
        Text(
          valor,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        Text(
          etiqueta,
          style: const TextStyle(fontSize: 12, color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildCondicionesPulverizacion() {
    final esApto = _weatherData!.esAptoParaPulverizar;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: esApto ? Colors.green[50] : Colors.orange[50],
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: esApto ? Colors.green : Colors.orange,
          width: 2,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                esApto ? Icons.check_circle : Icons.warning,
                color: esApto ? Colors.green : Colors.orange,
                size: 32,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Condiciones para Pulverización',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: esApto ? Colors.green[900] : Colors.orange[900],
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            _weatherData!.recomendacionPulverizacion,
            style: TextStyle(
              fontSize: 15,
              color: esApto ? Colors.green[800] : Colors.orange[800],
            ),
          ),
          const SizedBox(height: 16),
          _buildParametrosOptimos(),
        ],
      ),
    );
  }

  Widget _buildParametrosOptimos() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Parámetros óptimos:',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: Colors.grey[700],
          ),
        ),
        const SizedBox(height: 8),
        _buildParametro('Temperatura', '18-28°C', _weatherData!.temperatura),
        _buildParametro('Humedad', '50-80%', _weatherData!.humedadRelativa),
        _buildParametro('Viento', '< 15 km/h', _weatherData!.velocidadViento),
      ],
    );
  }

  Widget _buildParametro(String nombre, String rango, double valor) {
    bool dentroRango = false;
    if (nombre == 'Temperatura') {
      dentroRango = valor >= 18 && valor <= 28;
    } else if (nombre == 'Humedad') {
      dentroRango = valor >= 50 && valor <= 80;
    } else if (nombre == 'Viento') {
      dentroRango = valor < 15;
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Icon(
            dentroRango ? Icons.check : Icons.close,
            size: 16,
            color: dentroRango ? Colors.green : Colors.red,
          ),
          const SizedBox(width: 8),
          Text(
            '$nombre: $rango',
            style: TextStyle(fontSize: 13, color: Colors.grey[700]),
          ),
        ],
      ),
    );
  }

  Widget _buildPronosticoHorario() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Pronóstico por Hora',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 160,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: _weatherData!.pronosticoHorario.length,
            itemBuilder: (context, index) {
              final pronostico = _weatherData!.pronosticoHorario[index];
              return _buildTarjetaHora(pronostico);
            },
          ),
        ),
      ],
    );
  }

  Widget _buildTarjetaHora(PronosticoHora pronostico) {
    final hora = DateFormat('HH:mm').format(pronostico.hora);

    return Container(
      width: 100,
      margin: const EdgeInsets.only(right: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            hora,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(pronostico.iconoClima, style: const TextStyle(fontSize: 32)),
          const SizedBox(height: 8),
          Text(
            '${pronostico.temperatura.toStringAsFixed(0)}°C',
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.water_drop, size: 12, color: Colors.blue),
              const SizedBox(width: 2),
              Text(
                '${pronostico.probabilidadLluvia.toStringAsFixed(0)}%',
                style: const TextStyle(fontSize: 11, color: Colors.blue),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // ✨ NUEVA SECCIÓN: Detalles Meteorológicos
  Widget _buildDetallesMeteorologicos() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Detalles Meteorológicos',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                const Color.fromARGB(255, 141, 141, 217),
                const Color.fromARGB(255, 141, 141, 217),
                const Color.fromARGB(255, 141, 141, 217),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: Colors.indigo.withOpacity(0.3),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              IntrinsicHeight(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      child: _buildDetalleItem(
                        icon: Icons.wb_sunny,
                        titulo: 'Índice UV',
                        valor: _weatherData!.calidadAire,
                        subtitulo: _weatherData!.indiceUV.toString(),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildDetalleItem(
                        icon: Icons.water_drop,
                        titulo: 'Humedad',
                        valor:
                            '${_weatherData!.humedadRelativa.toStringAsFixed(0)}%',
                        subtitulo: _weatherData!.humedadRelativa > 70
                            ? 'Alta'
                            : _weatherData!.humedadRelativa < 40
                            ? 'Baja'
                            : 'Normal',
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              IntrinsicHeight(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      child: _buildDetalleItem(
                        icon: Icons.air,
                        titulo: 'Viento',
                        valor: _weatherData!.descripcionViento,
                        subtitulo:
                            '${_weatherData!.velocidadViento.toStringAsFixed(0)} km/h',
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildDetalleItem(
                        icon: Icons.thermostat,
                        titulo: 'Punto de rocío',
                        valor: _weatherData!.descripcionPuntoRocio,
                        subtitulo:
                            '${_weatherData!.puntoRocio.toStringAsFixed(0)}°',
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              IntrinsicHeight(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      child: _buildDetalleItem(
                        icon: Icons.compress,
                        titulo: 'Presión',
                        valor: _weatherData!.presion > 1013
                            ? 'Aumentando'
                            : _weatherData!.presion < 1013
                            ? 'Bajando'
                            : 'Estable',
                        subtitulo:
                            '${_weatherData!.presion.toStringAsFixed(1)} mb',
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildDetalleItem(
                        icon: Icons.visibility,
                        titulo: 'Visibilidad',
                        valor: _weatherData!.visibilidad > 10
                            ? 'Excelente'
                            : _weatherData!.visibilidad > 5
                            ? 'Buena'
                            : 'Reducida',
                        subtitulo:
                            '${_weatherData!.visibilidad.toStringAsFixed(2)} km',
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildDetalleItem({
    required IconData icon,
    required String titulo,
    required String valor,
    required String subtitulo,
  }) {
    Color colorIcono;

    switch (icon) {
      case Icons.wb_sunny:
        colorIcono = Colors.amberAccent;
        break;
      case Icons.water_drop:
        colorIcono = Colors.lightBlueAccent;
        break;
      case Icons.air:
        colorIcono = Colors.cyanAccent;
        break;
      case Icons.thermostat:
        colorIcono = Colors.redAccent;
        break;
      case Icons.compress:
        colorIcono = Colors.deepPurpleAccent;
        break;
      case Icons.visibility:
        colorIcono = Colors.tealAccent;
        break;
      default:
        colorIcono = Colors.white70;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            Icon(icon, color: colorIcono, size: 20),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                titulo,
                style: TextStyle(
                  color: colorIcono,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          valor,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 3),
        Text(
          subtitulo,
          style: TextStyle(color: Colors.white.withOpacity(0.75), fontSize: 12),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }

  Widget _buildBotonIA() {
    return Center(
      child: ElevatedButton.icon(
        onPressed: _cargandoIA ? null : _obtenerRecomendacionIA,
        icon: _cargandoIA
            ? const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              )
            : const Icon(Icons.psychology),
        label: Text(
          _cargandoIA ? 'Consultando IA...' : 'Obtener Recomendaciones IA',
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.deepPurple,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }

  Widget _buildRecomendacionesIA() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Colors.deepPurple[50]!, Colors.deepPurple[100]!],
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.deepPurple, width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: const BoxDecoration(
                  color: Colors.deepPurple,
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.psychology,
                  color: Colors.white,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Text(
                  'Recomendaciones IA',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.deepPurple,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            _recomendacionIA!,
            style: TextStyle(
              fontSize: 15,
              height: 1.5,
              color: Colors.deepPurple[900],
            ),
          ),
        ],
      ),
    );
  }
}
