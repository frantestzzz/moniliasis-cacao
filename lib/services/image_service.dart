// lib/services/image_service.dart
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:permission_handler/permission_handler.dart';
import 'dart:developer' as developer;

class ImageService {
  static final ImageService _instance = ImageService._internal();
  factory ImageService() => _instance;
  ImageService._internal();

  final ImagePicker _picker = ImagePicker();

  /// Solicitar permisos necesarios
  Future<bool> solicitarPermisos(ImageSource source) async {
    if (source == ImageSource.camera) {
      final cameraStatus = await Permission.camera.request();
      return cameraStatus.isGranted;
    } else {
      // Para galería en Android 13+
      if (Platform.isAndroid) {
        final photosStatus = await Permission.photos.request();
        if (photosStatus.isPermanentlyDenied) {
          await openAppSettings();
          return false;
        }
        return photosStatus.isGranted;
      }
      return true;
    }
  }

  /// Tomar foto desde cámara o galería
  Future<String?> seleccionarImagen({
    required ImageSource source,
    int maxWidth = 800,
    int maxHeight = 800,
    int quality = 85,
  }) async {
    try {
      // Solicitar permisos
      final hasPermission = await solicitarPermisos(source);
      if (!hasPermission) {
        developer.log(
          '❌ Permisos denegados para ${source == ImageSource.camera ? 'cámara' : 'galería'}',
          name: 'ImageService',
        );
        return null;
      }

      // Seleccionar imagen
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: maxWidth.toDouble(),
        maxHeight: maxHeight.toDouble(),
        imageQuality: quality,
      );

      if (image == null) {
        developer.log(
          '❌ No se seleccionó ninguna imagen',
          name: 'ImageService',
        );
        return null;
      }

      // Guardar imagen en directorio local
      final String? savedPath = await guardarImagenLocal(image.path);

      if (savedPath != null) {
        developer.log('✅ Imagen guardada en: $savedPath', name: 'ImageService');
      }

      return savedPath;
    } catch (e) {
      developer.log('❌ Error al seleccionar imagen: $e', name: 'ImageService');
      return null;
    }
  }

  /// Guardar imagen en directorio local permanente
  Future<String?> guardarImagenLocal(String imagePath) async {
    try {
      final Directory appDir = await getApplicationDocumentsDirectory();
      final String cultivosDir = path.join(appDir.path, 'cultivos_images');

      // Crear directorio si no existe
      final Directory dir = Directory(cultivosDir);
      if (!await dir.exists()) {
        await dir.create(recursive: true);
      }

      // Generar nombre único
      final String fileName = '${DateTime.now().millisecondsSinceEpoch}.jpg';
      final String newPath = path.join(cultivosDir, fileName);

      // Copiar archivo
      final File sourceFile = File(imagePath);
      await sourceFile.copy(newPath);

      return newPath;
    } catch (e) {
      developer.log('❌ Error al guardar imagen: $e', name: 'ImageService');
      return null;
    }
  }

  /// Eliminar imagen del almacenamiento local
  Future<bool> eliminarImagen(String? imagePath) async {
    if (imagePath == null || imagePath.isEmpty) return false;

    try {
      final File file = File(imagePath);
      if (await file.exists()) {
        await file.delete();
        developer.log('✅ Imagen eliminada: $imagePath', name: 'ImageService');
        return true;
      }
      return false;
    } catch (e) {
      developer.log('❌ Error al eliminar imagen: $e', name: 'ImageService');
      return false;
    }
  }

  /// Verificar si una imagen existe
  Future<bool> imagenExiste(String? imagePath) async {
    if (imagePath == null || imagePath.isEmpty) return false;

    try {
      final File file = File(imagePath);
      return await file.exists();
    } catch (e) {
      return false;
    }
  }
}
