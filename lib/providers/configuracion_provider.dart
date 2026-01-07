// lib/providers/configuracion_provider.dart
import 'package:flutter/material.dart';
import 'package:amgeca/Model/configuracion.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ConfiguracionProvider extends ChangeNotifier {
  Configuracion _configuracion = Configuracion();

  Configuracion get configuracion => _configuracion;

  String get tema => _configuracion.tema;
  String get tamanoFuente => _configuracion.tamanoFuente;
  bool get animaciones => _configuracion.animaciones;
  bool get vibracion => _configuracion.vibracion;

  // Cargar configuración al iniciar
  Future<void> cargarConfiguracion() async {
    final prefs = await SharedPreferences.getInstance();

    _configuracion = Configuracion(
      tema: prefs.getString('tema') ?? 'light',
      tamanoFuente: prefs.getString('tamanoFuente') ?? 'medium',
      animaciones: prefs.getBool('animaciones') ?? true,
      vibracion: prefs.getBool('vibracion') ?? true,
    );

    notifyListeners();
  }

  // Guardar configuración
  Future<void> guardarConfiguracion() async {
    final prefs = await SharedPreferences.getInstance();

    await prefs.setString('tema', _configuracion.tema);
    await prefs.setString('tamanoFuente', _configuracion.tamanoFuente);
    await prefs.setBool('animaciones', _configuracion.animaciones);
    await prefs.setBool('vibracion', _configuracion.vibracion);

    notifyListeners();
  }

  // Cambiar tema
  Future<void> cambiarTema(String nuevoTema) async {
    _configuracion = _configuracion.copyWith(tema: nuevoTema);
    await guardarConfiguracion();
  }

  // Cambiar tamaño de fuente
  Future<void> cambiarTamanoFuente(String nuevoTamano) async {
    _configuracion = _configuracion.copyWith(tamanoFuente: nuevoTamano);
    await guardarConfiguracion();
  }

  // Toggle animaciones
  Future<void> toggleAnimaciones() async {
    _configuracion = _configuracion.copyWith(
      animaciones: !_configuracion.animaciones,
    );
    await guardarConfiguracion();
  }

  // Toggle vibración
  Future<void> toggleVibracion() async {
    _configuracion = _configuracion.copyWith(
      vibracion: !_configuracion.vibracion,
    );
    await guardarConfiguracion();
  }

  // Obtener tamaño de fuente numérico
  double getTamanoFuenteNumerico() {
    switch (_configuracion.tamanoFuente) {
      case 'small':
        return 0.85;
      case 'large':
        return 1.15;
      case 'medium':
      default:
        return 1.0;
    }
  }
}
