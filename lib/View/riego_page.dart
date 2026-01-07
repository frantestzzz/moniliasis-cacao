import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:amgeca/providers/tasks_provider.dart';

class RiegoPage extends StatelessWidget {
  const RiegoPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Riegos programados'),
        backgroundColor: Colors.green[800],
      ),
      body: Consumer<TasksProvider>(
        builder: (context, provider, _) {
          final riegoTasks = provider.tasks
              .where((task) => task.category == 'Riego')
              .toList();
          if (riegoTasks.isEmpty) {
            return const Center(
              child: Text('No hay riegos programados para hoy.'),
            );
          }
          return ListView.separated(
            padding: const EdgeInsets.all(16),
            separatorBuilder: (_, __) => const Divider(),
            itemCount: riegoTasks.length,
            itemBuilder: (context, index) {
              final task = riegoTasks[index];
              return ListTile(
                leading: Icon(Icons.water_drop, color: Colors.blue[700]),
                title: Text(task.title),
                subtitle: Text(
                  '${task.cultivoName} • ${_formatDate(task.scheduledDate.toLocal())}',
                ),
                trailing: IconButton(
                  icon: Icon(
                    task.completed ? Icons.check_circle : Icons.water,
                    color: task.completed ? Colors.green : Colors.blue,
                  ),
                  tooltip: task.completed
                      ? 'Completado'
                      : 'Marcar como completado',
                  onPressed: task.completed
                      ? null
                      : () => provider.completeTask(task.id),
                ),
              );
            },
          );
        },
      ),
    );
  }

  String _formatDate(DateTime date) {
    final day = date.day.toString().padLeft(2, '0');
    final month = date.month.toString().padLeft(2, '0');
    final year = date.year;
    return '$day/$month/$year';
  }
}
