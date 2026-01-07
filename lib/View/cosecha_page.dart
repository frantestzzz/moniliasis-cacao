import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:amgeca/providers/crops_provider.dart';
import 'package:amgeca/classes/cultivo.dart';

class CosechaPage extends StatelessWidget {
  const CosechaPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cosechas próximas'),
        backgroundColor: Colors.green[800],
      ),
      body: Consumer<CropsProvider>(
        builder: (context, provider, _) {
          final upcoming = provider.upcomingHarvests();
          if (upcoming.isEmpty) {
            return const Center(
              child: Text('No hay cosechas próximas registradas.'),
            );
          }
          return ListView.separated(
            padding: const EdgeInsets.all(16),
            separatorBuilder: (_, __) => const Divider(),
            itemCount: upcoming.length,
            itemBuilder: (context, index) {
              final cultivo = upcoming[index];
              final fecha = DateTime.tryParse(cultivo.fechaCosecha ?? '');
              return ListTile(
                leading: const Icon(Icons.agriculture, color: Colors.green),
                title: Text(cultivo.nombre),
                subtitle: Text(
                  '${cultivo.tipoSuelo} • ${cultivo.area.toStringAsFixed(2)} ha • ${fecha != null ? _formatDate(fecha) : 'Sin fecha'}',
                ),
                trailing: ElevatedButton(
                  onPressed: () => _showRegisterDialog(context, cultivo),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green[800],
                  ),
                  child: const Text('Registrar'),
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

  void _showRegisterDialog(BuildContext context, Cultivo cultivo) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Registrar cosecha: ${cultivo.nombre}'),
        content: const Text('¿Confirmar que la cosecha ha sido completada?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancelar'),
          ),
          ElevatedButton(
            onPressed: () {
              // Aquí podrías llamar al provider para marcar como cosechado
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Cosecha registrada correctamente'),
                ),
              );
            },
            style: ElevatedButton.styleFrom(backgroundColor: Colors.green[800]),
            child: const Text('Confirmar'),
          ),
        ],
      ),
    );
  }
}
