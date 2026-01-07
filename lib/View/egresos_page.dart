import 'package:flutter/material.dart';
import 'package:amgeca/classes/egreso.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:intl/intl.dart';

class EgresosPage extends StatefulWidget {
  final Cultivo cultivo;

  const EgresosPage({Key? key, required this.cultivo}) : super(key: key);

  @override
  State<EgresosPage> createState() => _EgresosPageState();
}

class _EgresosPageState extends State<EgresosPage> {
  late Future<List<Egreso>> _egresosFuture;
  double _totalEgresos = 0.0;

  // Lista de tipos de egresos predefinidos
  final List<String> _tiposEgresos = [
    'Semillas',
    'Fertilizantes',
    'Pesticidas',
    'Herbicidas',
    'Mano de Obra',
    'Riego',
    'Mantenimiento',
    'Herramientas',
    'Transporte',
    'Almacenamiento',
    'Otros',
  ];

  @override
  void initState() {
    super.initState();
    _loadEgresos();
  }

  void _loadEgresos() {
    setState(() {
      _egresosFuture = _getEgresosFromDB();
    });
  }

  Future<List<Egreso>> _getEgresosFromDB() async {
    try {
      final rows = await BasedatoHelper.instance.getEgresosByCultivo(
        widget.cultivo.id!,
      );
      final egresos = rows.map((r) => Egreso.fromMap(r)).toList();

      // Calcular total
      _totalEgresos = egresos.fold(0.0, (sum, egreso) => sum + egreso.monto);

      return egresos;
    } catch (e) {
      print('Error al cargar egresos: $e');
      return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Egresos - ${widget.cultivo.nombre}'),
        backgroundColor: Colors.green[700],
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _mostrarFormularioEgreso(context),
            tooltip: 'Agregar Egreso',
          ),
        ],
      ),
      body: Column(
        children: [
          // Tarjeta de resumen
          _buildSummaryCard(),

          // Lista de egresos
          Expanded(
            child: FutureBuilder<List<Egreso>>(
              future: _egresosFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (snapshot.hasError) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.error, color: Colors.red[400], size: 48),
                        const SizedBox(height: 16),
                        Text(
                          'Error al cargar egresos',
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.grey[600],
                          ),
                        ),
                        const SizedBox(height: 8),
                        ElevatedButton(
                          onPressed: _loadEgresos,
                          child: const Text('Reintentar'),
                        ),
                      ],
                    ),
                  );
                }

                final egresos = snapshot.data ?? [];

                if (egresos.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.money_off,
                          color: Colors.grey[400],
                          size: 64,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'No hay egresos registrados',
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.grey[600],
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Agrega tu primer egreso para comenzar',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey[500],
                          ),
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          onPressed: () => _mostrarFormularioEgreso(context),
                          icon: const Icon(Icons.add),
                          label: const Text('Agregar Egreso'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green[700],
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  );
                }

                return RefreshIndicator(
                  onRefresh: () async {
                    _loadEgresos();
                  },
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: egresos.length,
                    itemBuilder: (context, index) {
                      final egreso = egresos[index];
                      return _buildEgresoCard(egreso);
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryCard() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.red[400]!, Colors.red[600]!],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.red.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.money_off, color: Colors.white, size: 28),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Total de Egresos',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'S/ ${_totalEgresos.toStringAsFixed(2)}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEgresoCard(Egreso egreso) {
    final tipoColor = _getTipoColor(egreso.tipo);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: tipoColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    _getTipoIcon(egreso.tipo),
                    color: tipoColor,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        egreso.tipo,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        egreso.descripcion,
                        style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      'S/ ${egreso.monto.toStringAsFixed(2)}',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.red[600],
                      ),
                    ),
                    Text(
                      DateFormat(
                        'dd/MM/yyyy',
                      ).format(DateTime.parse(egreso.fecha)),
                      style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                    ),
                  ],
                ),
              ],
            ),
            if (egreso.notas != null && egreso.notas!.isNotEmpty) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.grey[50],
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Row(
                  children: [
                    Icon(Icons.notes, size: 14, color: Colors.grey[600]),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        egreso.notas!,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[700],
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                IconButton(
                  icon: const Icon(Icons.edit, color: Colors.blue, size: 20),
                  onPressed: () => _editarEgreso(egreso),
                  tooltip: 'Editar',
                ),
                IconButton(
                  icon: const Icon(Icons.delete, color: Colors.red, size: 20),
                  onPressed: () => _eliminarEgreso(egreso),
                  tooltip: 'Eliminar',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Color _getTipoColor(String tipo) {
    switch (tipo) {
      case 'Semillas':
        return Colors.green;
      case 'Fertilizantes':
        return Colors.brown;
      case 'Pesticidas':
        return Colors.orange;
      case 'Herbicidas':
        return Colors.purple;
      case 'Mano de Obra':
        return Colors.blue;
      case 'Riego':
        return Colors.cyan;
      case 'Mantenimiento':
        return Colors.grey;
      case 'Herramientas':
        return Colors.amber;
      case 'Transporte':
        return Colors.indigo;
      case 'Almacenamiento':
        return Colors.teal;
      default:
        return Colors.red;
    }
  }

  IconData _getTipoIcon(String tipo) {
    switch (tipo) {
      case 'Semillas':
        return Icons.grain;
      case 'Fertilizantes':
        return Icons.eco;
      case 'Pesticidas':
        return Icons.bug_report;
      case 'Herbicidas':
        return Icons.grass;
      case 'Mano de Obra':
        return Icons.people;
      case 'Riego':
        return Icons.water_drop;
      case 'Mantenimiento':
        return Icons.build;
      case 'Herramientas':
        return Icons.hardware;
      case 'Transporte':
        return Icons.local_shipping;
      case 'Almacenamiento':
        return Icons.warehouse;
      default:
        return Icons.money_off;
    }
  }

  void _mostrarFormularioEgreso(BuildContext context, {Egreso? egreso}) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => FormularioEgreso(
        cultivo: widget.cultivo,
        egreso: egreso,
        tiposEgresos: _tiposEgresos,
        onSaved: () {
          Navigator.pop(context);
          _loadEgresos();
        },
      ),
    );
  }

  void _editarEgreso(Egreso egreso) {
    _mostrarFormularioEgreso(context, egreso: egreso);
  }

  void _eliminarEgreso(Egreso egreso) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Eliminar Egreso'),
        content: Text(
          '¿Estás seguro de eliminar el egreso de ${egreso.tipo} por S/ ${egreso.monto.toStringAsFixed(2)}?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Eliminar', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );

    if (confirm == true && egreso.id != null) {
      try {
        // Eliminar el egreso
        await BasedatoHelper.instance.deleteEgreso(egreso.id!);

        // Restar el monto del campo egresos del cultivo
        await _actualizarEgresosDelCultivo(egreso.monto, isIncrement: false);

        _loadEgresos();
        if (mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(const SnackBar(content: Text('Egreso eliminado')));
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error al eliminar egreso: $e')),
          );
        }
      }
    }
  }

  Future<void> _actualizarEgresosDelCultivo(
    double monto, {
    required bool isIncrement,
  }) async {
    try {
      // Obtener el cultivo actualizado
      final cultivosRows = await BasedatoHelper.instance.getAllCultivos();
      final cultivoActual = cultivosRows
          .map((r) => Cultivo.fromMap(r))
          .firstWhere((c) => c.id == widget.cultivo.id);

      // Calcular nuevo total de egresos
      double nuevoTotalEgresos;
      if (isIncrement) {
        nuevoTotalEgresos = (cultivoActual.egresos ?? 0.0) + monto;
      } else {
        nuevoTotalEgresos = (cultivoActual.egresos ?? 0.0) - monto;
        // Asegurar que no sea negativo
        nuevoTotalEgresos = nuevoTotalEgresos < 0 ? 0.0 : nuevoTotalEgresos;
      }

      // Actualizar el cultivo con el nuevo total de egresos
      await BasedatoHelper.instance.updateCultivo(widget.cultivo.id!, {
        'egresos': nuevoTotalEgresos,
      });
    } catch (e) {
      print('Error al actualizar egresos del cultivo: $e');
      // No mostrar error al usuario para no interrumpir el flujo principal
    }
  }
}

class FormularioEgreso extends StatefulWidget {
  final Cultivo cultivo;
  final Egreso? egreso;
  final List<String> tiposEgresos;
  final VoidCallback onSaved;

  const FormularioEgreso({
    Key? key,
    required this.cultivo,
    this.egreso,
    required this.tiposEgresos,
    required this.onSaved,
  }) : super(key: key);

  @override
  State<FormularioEgreso> createState() => _FormularioEgresoState();
}

class _FormularioEgresoState extends State<FormularioEgreso> {
  final _formKey = GlobalKey<FormState>();
  final _descripcionCtrl = TextEditingController();
  final _montoCtrl = TextEditingController();
  final _notasCtrl = TextEditingController();

  String _tipoSeleccionado = 'Semillas';
  DateTime _fechaSeleccionada = DateTime.now();

  @override
  void initState() {
    super.initState();
    if (widget.egreso != null) {
      _descripcionCtrl.text = widget.egreso!.descripcion;
      _montoCtrl.text = widget.egreso!.monto.toString();
      _notasCtrl.text = widget.egreso!.notas ?? '';
      _tipoSeleccionado = widget.egreso!.tipo;
      _fechaSeleccionada = DateTime.parse(widget.egreso!.fecha);
    }
  }

  @override
  void dispose() {
    _descripcionCtrl.dispose();
    _montoCtrl.dispose();
    _notasCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
        left: 16,
        right: 16,
        top: 16,
      ),
      child: Form(
        key: _formKey,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              widget.egreso == null ? 'Nuevo Egreso' : 'Editar Egreso',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),

            // Tipo de egreso
            DropdownButtonFormField<String>(
              value: _tipoSeleccionado,
              decoration: const InputDecoration(
                labelText: 'Tipo de Egreso',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.category),
              ),
              items: widget.tiposEgresos.map((tipo) {
                return DropdownMenuItem(value: tipo, child: Text(tipo));
              }).toList(),
              onChanged: (value) {
                setState(() {
                  _tipoSeleccionado = value!;
                });
              },
            ),
            const SizedBox(height: 12),

            // Descripción
            TextFormField(
              controller: _descripcionCtrl,
              decoration: const InputDecoration(
                labelText: 'Descripción',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.description),
              ),
              validator: (v) => (v == null || v.isEmpty) ? 'Requerido' : null,
              textInputAction: TextInputAction.next,
            ),
            const SizedBox(height: 12),

            // Monto
            TextFormField(
              controller: _montoCtrl,
              decoration: const InputDecoration(
                labelText: 'Monto (S/)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.money),
              ),
              keyboardType: TextInputType.number,
              validator: (v) {
                if (v == null || v.isEmpty) return 'Requerido';
                final monto = double.tryParse(v);
                if (monto == null) return 'Número inválido';
                if (monto <= 0) return 'El monto debe ser mayor a 0';
                return null;
              },
              textInputAction: TextInputAction.next,
            ),
            const SizedBox(height: 12),

            // Fecha
            InkWell(
              onTap: _seleccionarFecha,
              child: InputDecorator(
                decoration: const InputDecoration(
                  labelText: 'Fecha',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.calendar_today),
                ),
                child: Text(
                  DateFormat('dd/MM/yyyy').format(_fechaSeleccionada),
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Notas
            TextFormField(
              controller: _notasCtrl,
              decoration: const InputDecoration(
                labelText: 'Notas (opcional)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.notes),
              ),
              maxLines: 2,
            ),
            const SizedBox(height: 16),

            // Botones
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancelar'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _guardarEgreso,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red[600],
                      foregroundColor: Colors.white,
                    ),
                    child: const Text('Guardar'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Future<void> _seleccionarFecha() async {
    final fecha = await showDatePicker(
      context: context,
      initialDate: _fechaSeleccionada,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
    );
    if (fecha != null) {
      setState(() {
        _fechaSeleccionada = fecha;
      });
    }
  }

  Future<void> _guardarEgreso() async {
    if (!_formKey.currentState!.validate()) return;

    final monto = double.parse(_montoCtrl.text);

    final egreso = Egreso(
      id: widget.egreso?.id,
      cultivoId: widget.cultivo.id!,
      cultivoNombre: widget.cultivo.nombre,
      tipo: _tipoSeleccionado,
      descripcion: _descripcionCtrl.text,
      monto: monto,
      fecha: DateFormat('yyyy-MM-dd').format(_fechaSeleccionada),
      notas: _notasCtrl.text.isEmpty ? null : _notasCtrl.text,
    );

    try {
      if (widget.egreso == null) {
        // Es un nuevo egreso - insertar y actualizar cultivo
        print('Insertando nuevo egreso...');
        await BasedatoHelper.instance.insertEgreso(egreso.toMap());
        print('Egreso insertado correctamente');

        // Actualizar el campo egresos del cultivo
        print('Actualizando egresos del cultivo...');
        await _actualizarEgresosDelCultivo(monto, isIncrement: true);
        print('Egresos del cultivo actualizados');
      } else {
        // Es una edición - calcular diferencia y actualizar
        final diferencia = monto - widget.egreso!.monto;
        print('Editando egreso, diferencia: $diferencia');

        await BasedatoHelper.instance.updateEgreso(
          widget.egreso!.id!,
          egreso.toMap(),
        );
        print('Egreso actualizado correctamente');

        // Actualizar el campo egresos del cultivo con la diferencia
        await _actualizarEgresosDelCultivo(
          diferencia.abs(),
          isIncrement: diferencia > 0,
        );
        print('Egresos del cultivo actualizados con diferencia');
      }

      widget.onSaved();
    } catch (e) {
      print('Error completo al guardar egreso: $e');
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error al guardar egreso: $e')));
    }
  }

  Future<void> _actualizarEgresosDelCultivo(
    double monto, {
    required bool isIncrement,
  }) async {
    try {
      print('Buscando cultivo con ID: ${widget.cultivo.id}');

      // Obtener el cultivo actualizado
      final cultivosRows = await BasedatoHelper.instance.getAllCultivos();
      print('Se encontraron ${cultivosRows.length} cultivos en total');

      final cultivosList = cultivosRows.map((r) => Cultivo.fromMap(r)).toList();

      final cultivoActual = cultivosList
          .where((c) => c.id == widget.cultivo.id)
          .firstOrNull;

      if (cultivoActual == null) {
        print('ERROR: No se encontró el cultivo con ID ${widget.cultivo.id}');
        print('IDs disponibles: ${cultivosList.map((c) => c.id).toList()}');
        throw Exception('Cultivo no encontrado');
      }

      print(
        'Cultivo encontrado: ${cultivoActual.nombre}, egresos actuales: ${cultivoActual.egresos}',
      );

      // Calcular nuevo total de egresos
      double nuevoTotalEgresos;
      if (isIncrement) {
        nuevoTotalEgresos = (cultivoActual.egresos ?? 0.0) + monto;
      } else {
        nuevoTotalEgresos = (cultivoActual.egresos ?? 0.0) - monto;
        // Asegurar que no sea negativo
        nuevoTotalEgresos = nuevoTotalEgresos < 0 ? 0.0 : nuevoTotalEgresos;
      }

      print('Nuevo total de egresos: $nuevoTotalEgresos');

      // Actualizar el cultivo con el nuevo total de egresos
      await BasedatoHelper.instance.updateCultivo(widget.cultivo.id!, {
        'egresos': nuevoTotalEgresos,
      });

      print('Cultivo actualizado correctamente');
    } catch (e) {
      print('Error al actualizar egresos del cultivo: $e');
      // Relanzar la excepción para que se maneje en el método principal
      rethrow;
    }
  }
}
