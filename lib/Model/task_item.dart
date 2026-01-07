class TaskItem {
  final String id;
  final String title;
  final String description;
  final String cultivoName;
  final DateTime scheduledDate;
  final bool completed;
  final String category;

  TaskItem({
    required this.id,
    required this.title,
    required this.description,
    required this.cultivoName,
    required this.scheduledDate,
    this.completed = false,
    this.category = 'General',
  });
}
