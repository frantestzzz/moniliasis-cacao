// lib/Model/alert_item.dart
class AlertItem {
  final String id;
  final String title;
  final String message;
  final String severity; // 'critical', 'warning', 'info'
  final String cultivoName;
  final DateTime date;
  final String? targetRoute;
  final bool resuelta;

  AlertItem({
    required this.id,
    required this.title,
    required this.message,
    required this.severity,
    required this.cultivoName,
    required this.date,
    this.targetRoute,
    this.resuelta = false,
  });
}
