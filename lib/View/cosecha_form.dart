import 'package:flutter/material.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class CosechaFormPage extends StatefulWidget {
  final Cultivo cultivo;

  const CosechaFormPage({Key? key, required this.cultivo}) : super(key: key);

  @override
  State<CosechaFormPage> createState() => _CosechaFormPageState();
}

class _CosechaFormPageState extends State<CosechaFormPage> {
  late TextEditingController _cantidadCtrl;
  late TextEditingController _ingresosCtrl;
  late TextEditingController _egresosCtrl;
  late TextEditingController _egresosAdicionalesCtrl;
  late TextEditingController _fechaCosechaCtrl;
  late TextEditingController _notasCtrl;

  final _formKey = GlobalKey<FormState>();
  bool _isLoadingEgresos = true;
  double _totalEgresosCalculado = 0.0;

  // 🆕 Unidades de medida
  String _unidadSeleccionada = 'kg';
  final List<String> _unidadesMedida = ['kg', 'Toneladas', 'Sacos', 'Unidades'];

  @override
  void initState() {
    super.initState();
    _cantidadCtrl = TextEditingController(
      text: widget.cultivo.cantidadCosechada?.toString() ?? '',
    );
    _ingresosCtrl = TextEditingController(
      text: widget.cultivo.ingresos?.toString() ?? '',
    );
    _egresosCtrl = TextEditingController(
      text: widget.cultivo.egresos?.toString() ?? '',
    );
    _egresosAdicionalesCtrl = TextEditingController(); // 🆕 Campo adicional
    _fechaCosechaCtrl = TextEditingController(
      text: widget.cultivo.fechaCosecha ?? '',
    );
    _notasCtrl = TextEditingController(text: widget.cultivo.notas ?? '');

    // 🔥 Cargar egresos automáticamente
    _cargarEgresosAutomaticamente();

    // Listener para recalcular ganancia
    _ingresosCtrl.addListener(() => setState(() {}));
    _egresosAdicionalesCtrl.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _cantidadCtrl.dispose();
    _ingresosCtrl.dispose();
    _egresosCtrl.dispose();
    _egresosAdicionalesCtrl.dispose();
    _fechaCosechaCtrl.dispose();
    _notasCtrl.dispose();
    super.dispose();
  }

  // 🔥 CARGAR EGRESOS DESDE LA BASE DE DATOS
  Future<void> _cargarEgresosAutomaticamente() async {
    setState(() => _isLoadingEgresos = true);

    try {
      // Opción 1: Obtener del campo egresos del cultivo
      final cultivosRows = await BasedatoHelper.instance.getAllCultivos();
      final cultivoActual = cultivosRows
          .map((r) => Cultivo.fromMap(r))
          .firstWhere((c) => c.id == widget.cultivo.id);

      _totalEgresosCalculado = cultivoActual.egresos ?? 0.0;

      // Si el campo está vacío o es 0, calculamos desde la tabla egresos
      if (_totalEgresosCalculado == 0.0 && widget.cultivo.id != null) {
        _totalEgresosCalculado = await BasedatoHelper.instance
            .getEgresosTotalesPorCultivo(widget.cultivo.id!);
      }

      // Actualizar el campo de texto (no editable)
      _egresosCtrl.text = _totalEgresosCalculado.toStringAsFixed(2);

      print(
        '✅ Egresos cargados automáticamente: \$${_totalEgresosCalculado.toStringAsFixed(2)}',
      );
    } catch (e) {
      print('❌ Error al cargar egresos: $e');
      _totalEgresosCalculado = 0.0;
      _egresosCtrl.text = '0.00';
    } finally {
      setState(() => _isLoadingEgresos = false);
    }
  }

  // 💰 Calcular egresos totales (base + adicionales)
  double _calcularEgresosTotales() {
    final egresosBase = double.tryParse(_egresosCtrl.text) ?? 0.0;
    final egresosAdicionales =
        double.tryParse(_egresosAdicionalesCtrl.text) ?? 0.0;
    return egresosBase + egresosAdicionales;
  }

  Future<void> _selectDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(2020),
      lastDate: DateTime(2101),
    );
    if (picked != null) {
      setState(() {
        _fechaCosechaCtrl.text =
            '${picked.year}-${picked.month.toString().padLeft(2, '0')}-${picked.day.toString().padLeft(2, '0')}';
      });
    }
  }

  void _saveCosecha() async {
    if (_formKey.currentState!.validate()) {
      try {
        final egresosFinales = _calcularEgresosTotales();

        final cultivo = Cultivo(
          id: widget.cultivo.id,
          nombre: widget.cultivo.nombre,
          tipoSuelo: widget.cultivo.tipoSuelo,
          area: widget.cultivo.area,
          fechaSiembra: widget.cultivo.fechaSiembra,
          fechaCosecha: _fechaCosechaCtrl.text,
          estado: 'cosechado',
          notas: _notasCtrl.text,
          imagenUrl: widget.cultivo.imagenUrl,
          tipoId: widget.cultivo.tipoId,
          categoriaId: widget.cultivo.categoriaId,
          tipoRiego: widget.cultivo.tipoRiego,
          cantidadCosechada: double.tryParse(_cantidadCtrl.text),
          ingresos: double.tryParse(_ingresosCtrl.text),
          egresos:
              egresosFinales, // ✅ Guardamos egresos totales (base + adicionales)
        );

        await BasedatoHelper.instance.updateCultivo(
          cultivo.id!,
          cultivo.toMap(),
        );

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('✅ Cosecha registrada exitosamente'),
              backgroundColor: Colors.green,
              duration: Duration(seconds: 2),
            ),
          );
          Navigator.pop(context, true);
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('❌ Error al guardar: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final ingresos = double.tryParse(_ingresosCtrl.text) ?? 0.0;
    final egresosTotales = _calcularEgresosTotales();
    final ganancia = ingresos - egresosTotales;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Registrar Cosecha'),
        backgroundColor: Colors.green,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              // Información del cultivo
              Card(
                elevation: 3,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.cultivo.nombre,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Sembrío: ${widget.cultivo.fechaSiembra}',
                        style: TextStyle(color: Colors.grey[600]),
                      ),
                      Text(
                        'Área: ${widget.cultivo.area} ha',
                        style: TextStyle(color: Colors.grey[600]),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Cantidad cosechada
              TextFormField(
                controller: _cantidadCtrl,
                decoration: const InputDecoration(
                  labelText: 'Cantidad cosechada',
                  prefixIcon: Icon(Icons.agriculture),
                  border: OutlineInputBorder(),
                ),
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Ingresa la cantidad cosechada';
                  }
                  final cantidad = double.tryParse(value);
                  if (cantidad == null) {
                    return 'Ingresa un número válido';
                  }
                  if (cantidad <= 0) {
                    return 'La cantidad debe ser mayor a 0';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // 🆕 COMBO BOX DE UNIDADES
              DropdownButtonFormField<String>(
                value: _unidadSeleccionada,
                decoration: const InputDecoration(
                  labelText: 'Unidad de medida',
                  prefixIcon: Icon(Icons.straighten),
                  border: OutlineInputBorder(),
                ),
                items: _unidadesMedida.map((unidad) {
                  return DropdownMenuItem(value: unidad, child: Text(unidad));
                }).toList(),
                onChanged: (value) {
                  setState(() {
                    _unidadSeleccionada = value!;
                  });
                },
              ),
              const SizedBox(height: 16),

              // Fecha de cosecha
              TextFormField(
                controller: _fechaCosechaCtrl,
                readOnly: true,
                decoration: InputDecoration(
                  labelText: 'Fecha de cosecha',
                  prefixIcon: const Icon(Icons.calendar_today),
                  border: const OutlineInputBorder(),
                  suffixIcon: IconButton(
                    icon: const Icon(Icons.edit),
                    onPressed: _selectDate,
                  ),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Selecciona la fecha de cosecha';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Ingresos
              TextFormField(
                controller: _ingresosCtrl,
                decoration: const InputDecoration(
                  labelText: 'Ingresos (\$)',
                  prefixIcon: Icon(Icons.attach_money, color: Colors.green),
                  border: OutlineInputBorder(),
                ),
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Ingresa los ingresos';
                  }
                  if (double.tryParse(value) == null) {
                    return 'Ingresa un número válido';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // 💸 Egresos BASE (NO EDITABLE)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TextFormField(
                    controller: _egresosCtrl,
                    enabled: false, // ❌ NO EDITABLE
                    decoration: InputDecoration(
                      labelText: 'Egresos registrados (\$)',
                      prefixIcon: const Icon(
                        Icons.money_off,
                        color: Colors.red,
                      ),
                      border: const OutlineInputBorder(),
                      filled: true,
                      fillColor: Colors.grey[100],
                      suffixIcon: _isLoadingEgresos
                          ? const Padding(
                              padding: EdgeInsets.all(12),
                              child: SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              ),
                            )
                          : IconButton(
                              icon: const Icon(Icons.refresh, size: 20),
                              onPressed: _cargarEgresosAutomaticamente,
                              tooltip: 'Recargar egresos',
                            ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  // 📝 Nota informativa
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.blue[50],
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.blue[200]!),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.info_outline,
                          size: 16,
                          color: Colors.blue[700],
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Total de egresos ya registrados en el sistema',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.blue[700],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // 🆕 EGRESOS ADICIONALES (OPCIONAL)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TextFormField(
                    controller: _egresosAdicionalesCtrl,
                    decoration: InputDecoration(
                      labelText: 'Egresos adicionales (\$) - Opcional',
                      prefixIcon: const Icon(
                        Icons.add_circle,
                        color: Colors.orange,
                      ),
                      border: const OutlineInputBorder(),
                      helperText: 'Gastos extras no registrados previamente',
                      suffixIcon: _egresosAdicionalesCtrl.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear, size: 20),
                              onPressed: () {
                                setState(() {
                                  _egresosAdicionalesCtrl.clear();
                                });
                              },
                            )
                          : null,
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    validator: (value) {
                      if (value != null && value.isNotEmpty) {
                        if (double.tryParse(value) == null) {
                          return 'Ingresa un número válido';
                        }
                        if (double.parse(value) < 0) {
                          return 'No puede ser negativo';
                        }
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 8),
                  // 📊 Resumen de egresos
                  if (_egresosAdicionalesCtrl.text.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.orange[50],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.orange[200]!),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'Total de egresos:',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: Colors.orange[800],
                            ),
                          ),
                          Text(
                            '\$${egresosTotales.toStringAsFixed(2)}',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Colors.orange[900],
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 16),

              // Ganancia neta
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: ganancia >= 0 ? Colors.green[50] : Colors.red[50],
                  border: Border.all(
                    color: ganancia >= 0 ? Colors.green : Colors.red,
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Icon(
                          ganancia >= 0
                              ? Icons.trending_up
                              : Icons.trending_down,
                          color: ganancia >= 0 ? Colors.green : Colors.red,
                          size: 28,
                        ),
                        const SizedBox(width: 12),
                        const Text(
                          'Ganancia neta:',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                    Text(
                      '\$${ganancia.toStringAsFixed(2)}',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: ganancia >= 0
                            ? Colors.green[700]
                            : Colors.red[700],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Notas
              TextFormField(
                controller: _notasCtrl,
                decoration: const InputDecoration(
                  labelText: 'Notas adicionales',
                  prefixIcon: Icon(Icons.notes),
                  border: OutlineInputBorder(),
                  helperText: 'Información adicional sobre la cosecha',
                ),
                maxLines: 3,
              ),
              const SizedBox(height: 24),

              // Botón guardar
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _isLoadingEgresos ? null : _saveCosecha,
                  icon: const Icon(Icons.check_circle, size: 24),
                  label: const Text(
                    'Guardar Cosecha',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green[700],
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 3,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
