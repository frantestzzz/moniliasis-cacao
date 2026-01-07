import 'package:flutter/material.dart';
import 'package:amgeca/Model/task_item.dart';

class TasksProvider extends ChangeNotifier {
  final List<TaskItem> _tasks = [];

  List<TaskItem> get tasks => List.unmodifiable(_tasks);
  int get pendingCount => _tasks.where((task) => !task.completed).length;
  int get todayCount {
    final now = DateTime.now();
    return _tasks.where((task) {
      return !task.completed &&
          task.scheduledDate.year == now.year &&
          task.scheduledDate.month == now.month &&
          task.scheduledDate.day == now.day;
    }).length;
  }

  Future<void> loadMockTasks() async {
    _tasks
      ..clear()
      ..addAll([
        TaskItem(
          id: 't1',
          title: 'Aplicar fertilizante',
          description: 'Parcela 03 necesita potasio.',
          cultivoName: 'Tomate parcela 03',
          scheduledDate: DateTime.now(),
          category: 'Fertilización',
        ),
        TaskItem(
          id: 't2',
          title: 'Revisar riego',
          description: 'Verificar goteros en parcela 01.',
          cultivoName: 'Maíz parcela 01',
          scheduledDate: DateTime.now().add(const Duration(days: 1)),
          category: 'Riego',
        ),
        TaskItem(
          id: 't3',
          title: 'Evaluar plagas',
          description: 'Inspeccionar brotes del cultivo de frijol.',
          cultivoName: 'Frijol parcela 05',
          scheduledDate: DateTime.now().add(const Duration(days: 2)),
          category: 'Monitoreo',
        ),
      ]);
    notifyListeners();
  }

  void addTask(TaskItem task) {
    _tasks.add(task);
    notifyListeners();
  }

  void completeTask(String id) {
    final index = _tasks.indexWhere((task) => task.id == id);
    if (index != -1) {
      _tasks[index] = TaskItem(
        id: _tasks[index].id,
        title: _tasks[index].title,
        description: _tasks[index].description,
        cultivoName: _tasks[index].cultivoName,
        scheduledDate: _tasks[index].scheduledDate,
        completed: true,
        category: _tasks[index].category,
      );
      notifyListeners();
    }
  }
}
