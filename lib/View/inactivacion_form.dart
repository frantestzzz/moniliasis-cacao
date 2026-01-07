// lib/View/inactivacion_form.dart
import 'package:flutter/material.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class InactivacionFormPage extends StatefulWidget {
  final Cultivo cultivo;

  const InactivacionFormPage({Key? key, required this.cultivo})
    : super(key: key);

  @override
  State<InactivacionFormPage> createState() => _InactivacionFormPageState();
}

class _InactivacionFormPageState extends State<InactivacionFormPage> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _razonCtrl;
  String? _motivoInactivacion;

  final List<Map<String, dynamic>> _motivosInactivacion = [
    {
      'value': 'Pérdida total',
      'icon': Icons.dangerous,
      'color': Colors.red,
      'description': 'El cultivo se perdió completamente',
    },
    {
      'value': 'No rentable',
      'icon': Icons.money_off,
      'color': Colors.orange,
      'description': 'Los costos superan los beneficios',
    },
    {
      'value': 'Falta de recursos',
      'icon': Icons.water_damage,
      'color': Colors.blue,
      'description': 'No hay agua, insumos o mano de obra',
    },
    {
      'value': 'Cambio de cultivo',
      'icon': Icons.swap_horiz,
      'color': Colors.green,
      'description': 'Se decidió cultivar otra cosa',
    },
    {
      'value': 'Condiciones climáticas',
      'icon': Icons.wb_cloudy,
      'color': Colors.blueGrey,
      'description': 'Clima adverso permanente',
    },
    {
      'value': 'Abandono temporal',
      'icon': Icons.pause_circle,
      'color': Colors.grey,
      'description': 'Se suspende temporalmente',
    },
    {
      'value': 'Otro',
      'icon': Icons.more_horiz,
      'color': Colors.brown,
      'description': 'Otra razón',
    },
  ];

  @override
  void initState() {
    super.initState();
    _razonCtrl = TextEditingController();
  }

  @override
  void dispose() {
    _razonCtrl.dispose();
    super.dispose();
  }

  Future<void> _guardarInactivacion() async {
    if (_formKey.currentState!.validate()) {
      try {
        // Construir la nota combinada
        String notaFinal = widget.cultivo.notas ?? '';
        final fechaInactivacion = DateTime.now().toIso8601String().split(
          'T',
        )[0];
        final notaInactivacion =
            '\n\n🚫 INACTIVADO ($fechaInactivacion)\nMotivo: $_motivoInactivacion\nDetalle: ${_razonCtrl.text}';

        if (notaFinal.isNotEmpty) {
          notaFinal += notaInactivacion;
        } else {
          notaFinal = notaInactivacion.trim();
        }

        final cultivoActualizado = Cultivo(
          id: widget.cultivo.id,
          nombre: widget.cultivo.nombre,
          tipoSuelo: widget.cultivo.tipoSuelo,
          area: widget.cultivo.area,
          fechaSiembra: widget.cultivo.fechaSiembra,
          fechaCosecha: widget.cultivo.fechaCosecha,
          estado: Cultivo.ESTADO_INACTIVO,
          notas: notaFinal,
          imagenUrl: widget.cultivo.imagenUrl,
          tipoId: widget.cultivo.tipoId,
          categoriaId: widget.cultivo.categoriaId,
          tipoRiego: widget.cultivo.tipoRiego,
          cantidadCosechada: widget.cultivo.cantidadCosechada,
          ingresos: widget.cultivo.ingresos,
          egresos: widget.cultivo.egresos,
          razonRiesgo: widget.cultivo.razonRiesgo,
          tipoRiesgo: widget.cultivo.tipoRiesgo,
          fechaRiesgo: widget.cultivo.fechaRiesgo,
        );

        await BasedatoHelper.instance.updateCultivo(
          widget.cultivo.id!,
          cultivoActualizado.toMap(),
        );

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Row(
                children: [
                  Icon(Icons.check_circle, color: Colors.white),
                  SizedBox(width: 12),
                  Text('Cultivo inactivado exitosamente'),
                ],
              ),
              backgroundColor: Colors.grey,
            ),
          );
          Navigator.of(context).pop(true);
        }
      } catch (e) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Inactivar Cultivo'),
        backgroundColor: Colors.grey[700],
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Advertencia
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red, width: 2),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.warning_amber,
                      color: Colors.red,
                      size: 32,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            '¿Inactivar cultivo?',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Colors.red,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            widget.cultivo.nombre,
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Esta acción marcará el cultivo como inactivo. Puedes reactivarlo después si lo deseas.',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[700],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Motivo de inactivación
              const Text(
                'Motivo de Inactivación',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              ...(_motivosInactivacion.map((motivo) {
                final isSelected = _motivoInactivacion == motivo['value'];
                return Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: InkWell(
                    onTap: () {
                      setState(() {
                        _motivoInactivacion = motivo['value'];
                      });
                    },
                    borderRadius: BorderRadius.circular(12),
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? (motivo['color'] as Color).withOpacity(0.1)
                            : Colors.grey[100],
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: isSelected
                              ? motivo['color']
                              : Colors.grey[300]!,
                          width: isSelected ? 2 : 1,
                        ),
                      ),
                      child: Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: isSelected
                                  ? motivo['color']
                                  : Colors.grey[300],
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Icon(
                              motivo['icon'],
                              color: isSelected
                                  ? Colors.white
                                  : Colors.grey[600],
                              size: 24,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  motivo['value'],
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.bold,
                                    color: isSelected
                                        ? motivo['color']
                                        : Colors.black87,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  motivo['description'],
                                  style: TextStyle(
                                    fontSize: 11,
                                    color: Colors.grey[600],
                                  ),
                                ),
                              ],
                            ),
                          ),
                          if (isSelected)
                            Icon(
                              Icons.check_circle,
                              color: motivo['color'],
                              size: 24,
                            ),
                        ],
                      ),
                    ),
                  ),
                );
              }).toList()),
              const SizedBox(height: 24),

              // Detalles
              TextFormField(
                controller: _razonCtrl,
                decoration: InputDecoration(
                  labelText: 'Detalles de la Inactivación',
                  hintText: 'Explica con más detalle la razón...',
                  prefixIcon: Icon(Icons.description, color: Colors.grey[700]),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.grey[100],
                ),
                maxLines: 4,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Describe los detalles de la inactivación';
                  }
                  if (value.length < 15) {
                    return 'Proporciona más información (mínimo 15 caracteres)';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 24),

              // Información
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blue[200]!),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, color: Colors.blue[700]),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Esta información se guardará en las notas del cultivo para referencia futura.',
                        style: TextStyle(fontSize: 12, color: Colors.blue[900]),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Botones
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.close),
                      label: const Text('Cancelar'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.grey[700],
                        side: BorderSide(color: Colors.grey[400]!),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _motivoInactivacion == null
                          ? null
                          : _guardarInactivacion,
                      icon: const Icon(Icons.cancel),
                      label: const Text(
                        'Inactivar',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.grey[700],
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        disabledBackgroundColor: Colors.grey[300],
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
