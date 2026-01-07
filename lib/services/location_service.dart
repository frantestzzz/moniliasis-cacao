// lib/services/location_service.dart
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:geocoding/geocoding.dart';
import 'dart:developer' as developer;

class LocationService {
  static final LocationService _instance = LocationService._internal();
  factory LocationService() => _instance;
  LocationService._internal();

  Future<Position?> obtenerUbicacionActual() async {
    try {
      // Verificar si el servicio de ubicación está habilitado
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        developer.log(
          '❌ Servicio de ubicación deshabilitado',
          name: 'LocationService',
        );
        return null;
      }

      // Verificar permisos
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          developer.log(
            '❌ Permisos de ubicación denegados',
            name: 'LocationService',
          );
          return null;
        }
      }

      if (permission == LocationPermission.deniedForever) {
        developer.log(
          '❌ Permisos de ubicación denegados permanentemente',
          name: 'LocationService',
        );
        return null;
      }

      // Obtener ubicación
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: const Duration(seconds: 10),
      );

      developer.log(
        '✅ Ubicación obtenida: ${position.latitude}, ${position.longitude}',
        name: 'LocationService',
      );
      return position;
    } catch (e) {
      developer.log(
        '❌ Error al obtener ubicación: $e',
        name: 'LocationService',
      );
      return null;
    }
  }

  Future<bool> verificarYSolicitarPermisos() async {
    var status = await Permission.location.status;

    if (status.isDenied) {
      status = await Permission.location.request();
    }

    if (status.isPermanentlyDenied) {
      await openAppSettings();
      return false;
    }

    return status.isGranted;
  }

  Future<String> obtenerNombreCiudad(double lat, double lon) async {
    try {
      List<Placemark> placemarks = await placemarkFromCoordinates(lat, lon);

      if (placemarks.isNotEmpty) {
        Placemark place = placemarks[0];

        // Construir el nombre de la ubicación
        String ciudad = place.locality ?? place.subAdministrativeArea ?? '';
        String region = place.administrativeArea ?? '';
        String pais = place.country ?? '';

        // Retornar el formato más completo posible
        if (ciudad.isNotEmpty) {
          if (region.isNotEmpty) {
            return '$ciudad, $region';
          }
          return ciudad;
        } else if (region.isNotEmpty) {
          return region;
        } else if (pais.isNotEmpty) {
          return pais;
        }
      }

      // Si no se puede obtener el nombre, retornar coordenadas
      return '${lat.toStringAsFixed(4)}°, ${lon.toStringAsFixed(4)}°';
    } catch (e) {
      developer.log(
        '❌ Error al obtener nombre de ciudad: $e',
        name: 'LocationService',
      );
      return '${lat.toStringAsFixed(4)}°, ${lon.toStringAsFixed(4)}°';
    }
  }
}
