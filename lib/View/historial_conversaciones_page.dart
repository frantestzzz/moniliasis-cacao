// lib/View/historial_conversaciones_page.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../Data/basedato_helper.dart';

class HistorialConversacionesPage extends StatefulWidget {
  final Function(int conversacionId)? onSeleccionarConversacion;

  const HistorialConversacionesPage({Key? key, this.onSeleccionarConversacion})
    : super(key: key);

  @override
  State<HistorialConversacionesPage> createState() =>
      _HistorialConversacionesPageState();
}

class _HistorialConversacionesPageState
    extends State<HistorialConversacionesPage> {
  final BasedatoHelper db = BasedatoHelper.instance;
  List<Map<String, dynamic>> conversaciones = [];
  bool cargando = true;

  @override
  void initState() {
    super.initState();
    _cargarConversaciones();
  }

  Future<void> _cargarConversaciones() async {
    setState(() => cargando = true);

    final conversacionesDb = await db.getAllConversaciones();

    final conversacionesEnriquecidas = <Map<String, dynamic>>[];
    for (var conv in conversacionesDb) {
      final ultimoMensaje = await db.getUltimoMensaje(conv['id'] as int);

      conversacionesEnriquecidas.add({
        ...conv,
        'ultimoMensaje': ultimoMensaje?['contenido'] ?? '',
      });
    }

    setState(() {
      conversaciones = conversacionesEnriquecidas;
      cargando = false;
    });
  }

  Future<void> _eliminarConversacion(int id) async {
    final confirmar = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Eliminar conversación'),
        content: const Text(
          '¿Estás seguro de que quieres eliminar esta conversación? Esta acción no se puede deshacer.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Eliminar'),
          ),
        ],
      ),
    );

    if (confirmar == true) {
      await db.eliminarConversacion(id);
      _cargarConversaciones();

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Conversación eliminada'),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  Future<void> _renombrarConversacion(int id, String tituloActual) async {
    final controller = TextEditingController(text: tituloActual);

    final nuevoTitulo = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Renombrar conversación'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: 'Nuevo título',
            border: OutlineInputBorder(),
          ),
          maxLines: 2,
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Guardar'),
          ),
        ],
      ),
    );

    if (nuevoTitulo != null && nuevoTitulo.isNotEmpty) {
      await db.actualizarTituloConversacion(id, nuevoTitulo);
      _cargarConversaciones();

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Conversación renombrada'),
          backgroundColor: Colors.green,
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  String _formatearFecha(String fechaStr) {
    try {
      final fecha = DateTime.parse(fechaStr);
      final ahora = DateTime.now();
      final diferencia = ahora.difference(fecha);

      if (diferencia.inMinutes < 1) {
        return 'Hace un momento';
      } else if (diferencia.inHours < 1) {
        return 'Hace ${diferencia.inMinutes} min';
      } else if (diferencia.inHours < 24) {
        return 'Hace ${diferencia.inHours} h';
      } else if (diferencia.inDays < 7) {
        return 'Hace ${diferencia.inDays} días';
      } else {
        return DateFormat('dd/MM/yyyy').format(fecha);
      }
    } catch (e) {
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Historial de Conversaciones'),
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: cargando
          ? const Center(
              child: CircularProgressIndicator(color: Colors.deepPurple),
            )
          : conversaciones.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.chat_bubble_outline,
                    size: 80,
                    color: Colors.grey[300],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'No hay conversaciones guardadas',
                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.grey[600],
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Inicia una nueva conversación\ncon el asistente',
                    style: TextStyle(fontSize: 14, color: Colors.grey[500]),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            )
          : RefreshIndicator(
              onRefresh: _cargarConversaciones,
              color: Colors.deepPurple,
              child: ListView.builder(
                padding: const EdgeInsets.all(8),
                itemCount: conversaciones.length,
                itemBuilder: (context, index) {
                  final conv = conversaciones[index];
                  final id = conv['id'] as int;
                  final titulo = conv['titulo'] as String;
                  final ultimaActualizacion =
                      conv['ultimaActualizacion'] as String;
                  final mensajesCount = conv['mensajesCount'] as int;
                  final ultimoMensaje = conv['ultimoMensaje'] as String;

                  return Card(
                    margin: const EdgeInsets.symmetric(
                      vertical: 6,
                      horizontal: 8,
                    ),
                    elevation: 2,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: InkWell(
                      onTap: () {
                        if (widget.onSeleccionarConversacion != null) {
                          widget.onSeleccionarConversacion!(id);
                        }
                      },
                      borderRadius: BorderRadius.circular(12),
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.deepPurple[100],
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: const Icon(
                                Icons.chat_bubble,
                                color: Colors.deepPurple,
                                size: 28,
                              ),
                            ),
                            const SizedBox(width: 12),

                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    titulo,
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                    maxLines: 2,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  const SizedBox(height: 4),
                                  if (ultimoMensaje.isNotEmpty)
                                    Text(
                                      ultimoMensaje,
                                      style: TextStyle(
                                        fontSize: 14,
                                        color: Colors.grey[600],
                                      ),
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  const SizedBox(height: 6),
                                  Row(
                                    children: [
                                      Icon(
                                        Icons.access_time,
                                        size: 14,
                                        color: Colors.grey[500],
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        _formatearFecha(ultimaActualizacion),
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: Colors.grey[500],
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      Icon(
                                        Icons.message,
                                        size: 14,
                                        color: Colors.grey[500],
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        '$mensajesCount mensajes',
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: Colors.grey[500],
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),

                            PopupMenuButton<String>(
                              icon: const Icon(
                                Icons.more_vert,
                                color: Colors.grey,
                              ),
                              onSelected: (value) {
                                if (value == 'rename') {
                                  _renombrarConversacion(id, titulo);
                                } else if (value == 'delete') {
                                  _eliminarConversacion(id);
                                }
                              },
                              itemBuilder: (context) => [
                                const PopupMenuItem(
                                  value: 'rename',
                                  child: Row(
                                    children: [
                                      Icon(Icons.edit, size: 20),
                                      SizedBox(width: 8),
                                      Text('Renombrar'),
                                    ],
                                  ),
                                ),
                                const PopupMenuItem(
                                  value: 'delete',
                                  child: Row(
                                    children: [
                                      Icon(
                                        Icons.delete,
                                        size: 20,
                                        color: Colors.red,
                                      ),
                                      SizedBox(width: 8),
                                      Text(
                                        'Eliminar',
                                        style: TextStyle(color: Colors.red),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
    );
  }
}
