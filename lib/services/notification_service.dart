// lib/services/notification_service.dart
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:developer' as developer;

class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) return;

    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const DarwinInitializationSettings initializationSettingsIOS =
        DarwinInitializationSettings(
          requestAlertPermission: true,
          requestBadgePermission: true,
          requestSoundPermission: true,
        );

    const InitializationSettings initializationSettings =
        InitializationSettings(
          android: initializationSettingsAndroid,
          iOS: initializationSettingsIOS,
        );

    await _notificationsPlugin.initialize(
      initializationSettings,
      onDidReceiveNotificationResponse: (NotificationResponse response) {
        developer.log(
          'Notificación tocada: ${response.payload}',
          name: 'NotificationService',
        );
      },
    );

    // Solicitar permisos
    await _solicitarPermisos();

    _initialized = true;
    developer.log(
      '✅ Servicio de notificaciones inicializado',
      name: 'NotificationService',
    );
  }

  Future<void> _solicitarPermisos() async {
    if (await Permission.notification.isDenied) {
      await Permission.notification.request();
    }

    // Para Android 13+ necesitamos el permiso específico
    final status = await Permission.notification.status;
    if (status.isDenied) {
      developer.log(
        '⚠️ Permisos de notificación denegados',
        name: 'NotificationService',
        level: 900, // WARNING level
      );
    }
  }

  Future<void> mostrarNotificacionClimaDesfavorable({
    required String titulo,
    required String mensaje,
    List<String>? problemas,
  }) async {
    await initialize();

    String cuerpo = mensaje;
    if (problemas != null && problemas.isNotEmpty) {
      cuerpo += '\n\n⚠️ Problemas:\n${problemas.map((p) => '• $p').join('\n')}';
    }

    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
          'clima_alertas',
          'Alertas de Clima',
          channelDescription: 'Notificaciones sobre condiciones climáticas',
          importance: Importance.high,
          priority: Priority.high,
          icon: '@mipmap/ic_launcher',
          styleInformation: BigTextStyleInformation(''),
        );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const NotificationDetails platformDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notificationsPlugin.show(
      0,
      titulo,
      cuerpo,
      platformDetails,
      payload: 'clima_desfavorable',
    );

    developer.log(
      '📢 Notificación mostrada: $titulo',
      name: 'NotificationService',
    );
  }

  Future<void> mostrarNotificacionClimaFavorable({
    required String titulo,
    required String mensaje,
  }) async {
    await initialize();

    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
          'clima_favorables',
          'Clima Favorable',
          channelDescription:
              'Notificaciones sobre condiciones climáticas favorables',
          importance: Importance.defaultImportance,
          priority: Priority.defaultPriority,
          icon: '@mipmap/ic_launcher',
        );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const NotificationDetails platformDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notificationsPlugin.show(
      1,
      titulo,
      mensaje,
      platformDetails,
      payload: 'clima_favorable',
    );

    developer.log(
      '📢 Notificación mostrada: $titulo',
      name: 'NotificationService',
    );
  }

  Future<void> cancelarTodasLasNotificaciones() async {
    await _notificationsPlugin.cancelAll();
  }
}
