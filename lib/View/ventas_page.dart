// lib/View/ventas_page.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:amgeca/classes/venta.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:intl/intl.dart';

class VentasPage extends StatefulWidget {
  const VentasPage({Key? key}) : super(key: key);

  @override
  State<VentasPage> createState() => _VentasPageState();
}

class _VentasPageState extends State<VentasPage> {
  late Future<List<Venta>> _ventasFuture;
  late Future<double> _totalVentasFuture;

  @override
  void initState() {
    super.initState();
    _loadVentas();
  }

  void _loadVentas() {
    setState(() {
      _ventasFuture = _getVentasFromDB();
      _totalVentasFuture = BasedatoHelper.instance.getTotalVentas();
    });
  }

  Future<List<Venta>> _getVentasFromDB() async {
    final rows = await BasedatoHelper.instance.getAllVentas();
    return rows.map((r) => Venta.fromMap(r)).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ventas', style: TextStyle(color: Colors.white)),
        backgroundColor: Colors.green[800],
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.analytics, color: Colors.white),
            onPressed: () {
              Navigator.pushNamed(context, '/reportes-ventas');
            },
            tooltip: 'Ver estadísticas',
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final horizontalPadding = constraints.maxWidth >= 640 ? 24.0 : 16.0;
          return Column(
            children: [
              // Header con resumen
              Padding(
                padding: EdgeInsets.symmetric(
                  horizontal: horizontalPadding,
                  vertical: 12,
                ),
                child: FutureBuilder<double>(
                  future: _totalVentasFuture,
                  builder: (context, snapshot) {
                    final totalVentas = snapshot.data ?? 0.0;
                    return Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [Colors.green[700]!, Colors.green[500]!],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: FutureBuilder<List<Venta>>(
                        future: _ventasFuture,
                        builder: (context, ventasSnapshot) {
                          final ventasCount = ventasSnapshot.data?.length ?? 0;
                          return Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Total de Ventas',
                                style: TextStyle(
                                  color: Colors.white70,
                                  fontSize: 14,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'S/ ${totalVentas.toStringAsFixed(2)}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 32,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                '$ventasCount transacciones registradas',
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                    );
                  },
                ),
              ),

              // Lista de ventas
              Expanded(
                child: Padding(
                  padding: EdgeInsets.symmetric(horizontal: horizontalPadding),
                  child: FutureBuilder<List<Venta>>(
                    future: _ventasFuture,
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const Center(child: CircularProgressIndicator());
                      }

                      if (snapshot.hasError) {
                        return Center(child: Text('Error: ${snapshot.error}'));
                      }

                      final ventas = snapshot.data ?? [];

                      if (ventas.isEmpty) {
                        return Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.shopping_cart_outlined,
                                size: 80,
                                color: Colors.grey[300],
                              ),
                              const SizedBox(height: 16),
                              Text(
                                'No hay ventas registradas',
                                style: TextStyle(
                                  color: Colors.grey[600],
                                  fontSize: 16,
                                ),
                              ),
                              const SizedBox(height: 8),
                              const Text(
                                'Presiona + para agregar una venta',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        );
                      }

                      return LayoutBuilder(
                        builder: (context, listConstraints) {
                          final crossAxisCount =
                              (listConstraints.maxWidth / 360).floor().clamp(
                                1,
                                3,
                              );
                          final isGrid = crossAxisCount > 1;

                          if (isGrid) {
                            return GridView.builder(
                              padding: const EdgeInsets.only(
                                top: 8,
                                bottom: 16,
                              ),
                              gridDelegate:
                                  SliverGridDelegateWithFixedCrossAxisCount(
                                    crossAxisCount: crossAxisCount,
                                    crossAxisSpacing: 16,
                                    mainAxisSpacing: 16,
                                    childAspectRatio: 1.2,
                                  ),
                              itemCount: ventas.length,
                              itemBuilder: (context, index) {
                                return _buildVentaCard(ventas[index]);
                              },
                            );
                          }

                          return ListView.builder(
                            padding: const EdgeInsets.only(top: 8, bottom: 16),
                            itemCount: ventas.length,
                            itemBuilder: (context, index) {
                              return _buildVentaCard(ventas[index]);
                            },
                          );
                        },
                      );
                    },
                  ),
                ),
              ),
            ],
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _mostrarFormularioVenta(context),
        backgroundColor: Colors.green[700],
        icon: const Icon(Icons.add),
        label: const Text('Nueva Venta'),
      ),
    );
  }

  Widget _buildVentaCard(Venta venta) {
    final fecha = DateFormat('dd/MM/yyyy').format(DateTime.parse(venta.fecha));

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        onTap: () => _verDetalleVenta(venta),
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.green[100],
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Icon(
                          Icons.shopping_bag,
                          color: Colors.green[700],
                          size: 24,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            venta.cultivoNombre,
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            venta.cliente,
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  Text(
                    'S/ ${venta.total.toStringAsFixed(2)}',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.green[700],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildInfoChip(
                    '${venta.cantidad} ${venta.unidad}',
                    Icons.scale,
                  ),
                  _buildInfoChip(
                    'S/ ${venta.precioUnitario}/${venta.unidad}',
                    Icons.payments,
                  ),
                  _buildInfoChip(fecha, Icons.calendar_today),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoChip(String text, IconData icon) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: Colors.grey[700]),
          const SizedBox(width: 4),
          Text(text, style: TextStyle(fontSize: 11, color: Colors.grey[700])),
        ],
      ),
    );
  }

  Future<void> _mostrarFormularioVenta(BuildContext context) async {
    final result = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => const FormularioVenta(),
    );

    if (result == true) {
      _loadVentas();
    }
  }

  void _verDetalleVenta(Venta venta) {
    final fecha = DateFormat('dd/MM/yyyy').format(DateTime.parse(venta.fecha));

    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Container(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Detalle de Venta',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
              const Divider(),
              const SizedBox(height: 16),
              _buildDetailRow('Cultivo', venta.cultivoNombre),
              _buildDetailRow('Cliente', venta.cliente),
              _buildDetailRow('Cantidad', '${venta.cantidad} ${venta.unidad}'),
              _buildDetailRow('Precio unitario', 'S/ ${venta.precioUnitario}'),
              _buildDetailRow('Fecha', fecha),
              if (venta.notas != null && venta.notas!.isNotEmpty)
                _buildDetailRow('Notas', venta.notas!),
              const Divider(),
              _buildDetailRow(
                'Total',
                'S/ ${venta.total.toStringAsFixed(2)}',
                isBold: true,
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () {
                        Navigator.pop(context);
                        _editarVenta(venta);
                      },
                      icon: const Icon(Icons.edit),
                      label: const Text('Editar'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.pop(context);
                        _eliminarVenta(venta);
                      },
                      icon: const Icon(Icons.delete),
                      label: const Text('Eliminar'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.red,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildDetailRow(String label, String value, {bool isBold = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[600], fontSize: 14)),
          Flexible(
            child: Text(
              value,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
              ),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _editarVenta(Venta venta) async {
    final result = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => FormularioVenta(venta: venta),
    );

    if (result == true) {
      _loadVentas();
    }
  }

  Future<void> _eliminarVenta(Venta venta) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Eliminar Venta'),
        content: Text(
          '¿Estás seguro de eliminar la venta de ${venta.cultivoNombre}?',
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

    if (confirm == true && venta.id != null) {
      await BasedatoHelper.instance.deleteVenta(venta.id!);
      _loadVentas();
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Venta eliminada')));
      }
    }
  }
}

// Formulario de Venta
class FormularioVenta extends StatefulWidget {
  final Venta? venta;
  const FormularioVenta({Key? key, this.venta}) : super(key: key);

  @override
  State<FormularioVenta> createState() => _FormularioVentaState();
}

class _FormularioVentaState extends State<FormularioVenta> {
  final _formKey = GlobalKey<FormState>();
  final _cantidadCtrl = TextEditingController();
  final _precioCtrl = TextEditingController();
  final _clienteCtrl = TextEditingController();
  final _notasCtrl = TextEditingController();

  Cultivo? _cultivoSeleccionado;
  String _unidadSeleccionada = 'kg';
  List<Cultivo> _cultivos = [];
  List<Map<String, dynamic>> _tiposCultivo = [];
  String? _tipoSeleccionadoId;

  final List<String> _unidades = ['kg', 'toneladas', 'sacos', 'unidades'];

  @override
  void initState() {
    super.initState();
    _loadCultivos();
    if (widget.venta != null) {
      _cantidadCtrl.text = widget.venta!.cantidad.toString();
      _precioCtrl.text = widget.venta!.precioUnitario.toString();
      _clienteCtrl.text = widget.venta!.cliente;
      _notasCtrl.text = widget.venta!.notas ?? '';
      _unidadSeleccionada = widget.venta!.unidad;
    }
  }

  Future<void> _loadCultivos() async {
    final rows = await BasedatoHelper.instance.getAllCultivos();
    final tiposRows = await BasedatoHelper.instance.getAllTiposCultivo();

    setState(() {
      _cultivos = rows.map((r) => Cultivo.fromMap(r)).toList();
      _tiposCultivo = tiposRows;

      if (widget.venta != null) {
        _cultivoSeleccionado = _cultivos.firstWhere(
          (c) => c.id == widget.venta!.cultivoId,
          orElse: () => _cultivos.first,
        );

        // Seleccionar el tipo correspondiente al cultivo
        if (_cultivoSeleccionado != null &&
            _cultivoSeleccionado!.tipoId != null) {
          _tipoSeleccionadoId = _cultivoSeleccionado!.tipoId.toString();
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final horizontalPadding = constraints.maxWidth >= 640 ? 32.0 : 16.0;
        final isWide = constraints.maxWidth >= 640;
        return Padding(
          padding: EdgeInsets.only(
            bottom: MediaQuery.of(context).viewInsets.bottom,
            left: horizontalPadding,
            right: horizontalPadding,
            top: 16,
          ),
          child: Form(
            key: _formKey,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    widget.venta == null ? 'Nueva Venta' : 'Editar Venta',
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Selector de tipo de cultivo
                  DropdownButtonFormField<String>(
                    value: _tipoSeleccionadoId,
                    decoration: const InputDecoration(
                      labelText: 'Tipo de Cultivo',
                      border: OutlineInputBorder(),
                      prefixIcon: Icon(Icons.agriculture),
                    ),
                    items: _tiposCultivo.map((tipo) {
                      return DropdownMenuItem(
                        value: tipo['id']?.toString(),
                        child: Text(tipo['nombre']?.toString() ?? 'Sin nombre'),
                      );
                    }).toList(),
                    onChanged: (value) {
                      setState(() {
                        _tipoSeleccionadoId = value;
                        // Cuando se selecciona un tipo, buscar un cultivo de ese tipo
                        if (value != null) {
                          final cultivosDeTipo = _cultivos
                              .where((c) => c.tipoId.toString() == value)
                              .toList();
                          if (cultivosDeTipo.isNotEmpty) {
                            _cultivoSeleccionado = cultivosDeTipo.first;
                          } else {
                            // Si no hay cultivos de este tipo, usar el primer cultivo disponible
                            if (_cultivos.isNotEmpty) {
                              _cultivoSeleccionado = _cultivos.first;
                            }
                          }
                        }
                      });
                    },
                    validator: (value) =>
                        value == null ? 'Selecciona un tipo de cultivo' : null,
                  ),
                  const SizedBox(height: 12),

                  // Cliente
                  TextFormField(
                    controller: _clienteCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Cliente',
                      border: OutlineInputBorder(),
                      prefixIcon: Icon(Icons.person),
                    ),
                    inputFormatters: [
                      FilteringTextInputFormatter.allow(
                        RegExp(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]'),
                      ),
                    ],
                    validator: (v) {
                      if (v == null || v.isEmpty) return 'Requerido';
                      // Validar que solo contenga letras y espacios
                      if (!RegExp(r'^[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+$').hasMatch(v)) {
                        return 'Solo se permiten letras y espacios';
                      }
                      if (v.trim().length < 3) return 'Mínimo 3 caracteres';
                      return null;
                    },
                    textInputAction: TextInputAction.next,
                  ),
                  const SizedBox(height: 12),

                  // Cantidad y unidad
                  _buildCantidadUnidad(isWide),
                  const SizedBox(height: 12),

                  // Precio unitario
                  TextFormField(
                    controller: _precioCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Precio por unidad (S/)',
                      border: OutlineInputBorder(),
                      prefixIcon: Icon(Icons.payments),
                    ),
                    keyboardType: TextInputType.number,
                    validator: (v) {
                      if (v == null || v.isEmpty) return 'Requerido';
                      final precio = double.tryParse(v);
                      if (precio == null) return 'Número inválido';
                      if (precio <= 0) return 'El precio debe ser mayor a 0';
                      return null;
                    },
                    textInputAction: TextInputAction.next,
                  ),
                  const SizedBox(height: 12),

                  // Fecha (solo lectura, fecha actual automática)
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.calendar_today,
                              color: Colors.grey[600],
                              size: 20,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Fecha de Venta',
                              style: TextStyle(
                                color: Colors.grey[600],
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            Icon(Icons.lock, color: Colors.grey[400], size: 16),
                            const SizedBox(width: 8),
                            Text(
                              DateFormat('dd/MM/yyyy').format(DateTime.now()),
                              style: const TextStyle(fontSize: 16),
                            ),
                            const Spacer(),
                            Text(
                              'Actual',
                              style: TextStyle(
                                color: Colors.grey[600],
                                fontSize: 12,
                                fontStyle: FontStyle.italic,
                              ),
                            ),
                          ],
                        ),
                      ],
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
                  _buildActionButtons(isWide),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildCantidadUnidad(bool isWide) {
    if (isWide) {
      return Row(
        children: [
          Expanded(
            flex: 2,
            child: TextFormField(
              controller: _cantidadCtrl,
              decoration: const InputDecoration(
                labelText: 'Cantidad',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.scale),
              ),
              keyboardType: TextInputType.number,
              validator: (v) {
                if (v == null || v.isEmpty) return 'Requerido';
                final cantidad = double.tryParse(v);
                if (cantidad == null) return 'Número inválido';
                if (cantidad <= 0) return 'La cantidad debe ser mayor a 0';
                return null;
              },
              textInputAction: TextInputAction.next,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: DropdownButtonFormField<String>(
              value: _unidadSeleccionada,
              decoration: const InputDecoration(
                labelText: 'Unidad',
                border: OutlineInputBorder(),
              ),
              items: _unidades.map((unidad) {
                return DropdownMenuItem(value: unidad, child: Text(unidad));
              }).toList(),
              onChanged: (value) {
                setState(() {
                  _unidadSeleccionada = value!;
                });
              },
            ),
          ),
        ],
      );
    } else {
      return Column(
        children: [
          TextFormField(
            controller: _cantidadCtrl,
            decoration: const InputDecoration(
              labelText: 'Cantidad',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.scale),
            ),
            keyboardType: TextInputType.number,
            validator: (v) {
              if (v == null || v.isEmpty) return 'Requerido';
              final cantidad = double.tryParse(v);
              if (cantidad == null) return 'Número inválido';
              if (cantidad <= 0) return 'La cantidad debe ser mayor a 0';
              return null;
            },
            textInputAction: TextInputAction.next,
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _unidadSeleccionada,
            decoration: const InputDecoration(
              labelText: 'Unidad',
              border: OutlineInputBorder(),
            ),
            items: _unidades.map((unidad) {
              return DropdownMenuItem(value: unidad, child: Text(unidad));
            }).toList(),
            onChanged: (value) {
              setState(() {
                _unidadSeleccionada = value!;
              });
            },
          ),
        ],
      );
    }
  }

  Widget _buildActionButtons(bool isWide) {
    if (isWide) {
      return Row(
        children: [
          Expanded(
            child: OutlinedButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancelar'),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: ElevatedButton(
              onPressed: _guardarVenta,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green[700],
              ),
              child: const Text('Guardar'),
            ),
          ),
        ],
      );
    } else {
      return Column(
        children: [
          OutlinedButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancelar'),
          ),
          const SizedBox(height: 12),
          ElevatedButton(
            onPressed: _guardarVenta,
            style: ElevatedButton.styleFrom(backgroundColor: Colors.green[700]),
            child: const Text('Guardar'),
          ),
        ],
      );
    }
  }

  Future<void> _guardarVenta() async {
    if (!_formKey.currentState!.validate()) return;
    if (_tipoSeleccionadoId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Por favor selecciona un tipo de cultivo'),
        ),
      );
      return;
    }

    final cantidad = double.parse(_cantidadCtrl.text);
    final precio = double.parse(_precioCtrl.text);
    final total = cantidad * precio;

    // Obtener el nombre del tipo seleccionado
    final tipoSeleccionado = _tiposCultivo.firstWhere(
      (tipo) => tipo['id']?.toString() == _tipoSeleccionadoId,
      orElse: () => {},
    );

    // Si no hay cultivos de este tipo, crear uno temporal o usar el primer cultivo disponible
    if (_cultivoSeleccionado == null) {
      final cultivosDeTipo = _cultivos
          .where((c) => c.tipoId.toString() == _tipoSeleccionadoId)
          .toList();

      if (cultivosDeTipo.isEmpty) {
        // Si no hay cultivos de este tipo, usar el primer cultivo disponible
        if (_cultivos.isNotEmpty) {
          _cultivoSeleccionado = _cultivos.first;
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No hay cultivos disponibles')),
          );
          return;
        }
      } else {
        _cultivoSeleccionado = cultivosDeTipo.first;
      }
    }

    final venta = Venta(
      id: widget.venta?.id,
      cultivoId: _cultivoSeleccionado!.id!,
      cultivoNombre: tipoSeleccionado['nombre']?.toString() ?? 'Sin tipo',
      cantidad: cantidad,
      unidad: _unidadSeleccionada,
      precioUnitario: precio,
      total: total,
      cliente: _clienteCtrl.text,
      fecha: DateFormat(
        'yyyy-MM-dd',
      ).format(DateTime.now()), // Usa fecha actual
      notas: _notasCtrl.text.isEmpty ? null : _notasCtrl.text,
    );

    try {
      if (widget.venta == null) {
        await BasedatoHelper.instance.insertVenta(venta.toMap());
      } else {
        await BasedatoHelper.instance.updateVenta(
          widget.venta!.id!,
          venta.toMap(),
        );
      }

      if (mounted) {
        Navigator.pop(context, true);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              widget.venta == null ? 'Venta registrada' : 'Venta actualizada',
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }
}
