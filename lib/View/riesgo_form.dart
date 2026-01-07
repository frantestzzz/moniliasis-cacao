// lib/View/riesgo_form.dart
import 'package:flutter/material.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class RiesgoFormPage extends StatefulWidget {
  final Cultivo cultivo;

  const RiesgoFormPage({Key? key, required this.cultivo}) : super(key: key);

  @override
  State<RiesgoFormPage> createState() => _RiesgoFormPageState();
}

class _RiesgoFormPageState extends State<RiesgoFormPage> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _razonCtrl;
  String? _tipoRiesgo;
  late TextEditingController _detallesCtrl;
  DateTime? _fechaInicioRiesgo;
  String? _nivelGravedad;

  final List<Map<String, dynamic>> _tiposRiesgo = [
    {
      'value': 'Climático',
      'icon': Icons.cloud,
      'color': Colors.blue,
      'description': 'Problemas por clima adverso',
    },
    {
      'value': 'Plagas',
      'icon': Icons.bug_report,
      'color': Colors.red,
      'description': 'Ataque de insectos o animales',
    },
    {
      'value': 'Enfermedades',
      'icon': Icons.coronavirus,
      'color': Colors.purple,
      'description': 'Hongos, bacterias o virus',
    },
    {
      'value': 'Suelo',
      'icon': Icons.landscape,
      'color': Colors.brown,
      'description': 'Problemas con la tierra',
    },
    {
      'value': 'Falta de agua',
      'icon': Icons.water_drop,
      'color': Colors.cyan,
      'description': 'Sequía o riego insuficiente',
    },
    {
      'value': 'Nutrición',
      'icon': Icons.eco,
      'color': Colors.green,
      'description': 'Deficiencia de nutrientes',
    },
    {
      'value': 'Otro',
      'icon': Icons.more_horiz,
      'color': Colors.grey,
      'description': 'Otro tipo de problema',
    },
  ];

  final List<Map<String, dynamic>> _nivelesGravedad = [
    {
      'value': 'Leve',
      'icon': Icons.sentiment_satisfied,
      'color': Colors.green,
      'description': 'Riesgo menor, fácil de controlar',
    },
    {
      'value': 'Moderado',
      'icon': Icons.sentiment_neutral,
      'color': Colors.orange,
      'description': 'Riesgo medio, requiere atención',
    },
    {
      'value': 'Grave',
      'icon': Icons.sentiment_very_dissatisfied,
      'color': Colors.red,
      'description': 'Riesgo alto, acción inmediata',
    },
  ];

  @override
  void initState() {
    super.initState();
    _razonCtrl = TextEditingController(text: widget.cultivo.razonRiesgo ?? '');
    _tipoRiesgo = widget.cultivo.tipoRiesgo;
    _detallesCtrl = TextEditingController();
  }

  @override
  void dispose() {
    _razonCtrl.dispose();
    _detallesCtrl.dispose();
    super.dispose();
  }

  Future<void> _guardarRiesgo() async {
    if (_formKey.currentState!.validate()) {
      // Validar campos adicionales
      if (_fechaInicioRiesgo == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Row(
              children: [
                Icon(Icons.calendar_today, color: Colors.white),
                SizedBox(width: 12),
                Text('Selecciona la fecha de inicio del riesgo'),
              ],
            ),
            backgroundColor: Colors.orange,
          ),
        );
        return;
      }

      if (_nivelGravedad == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Row(
              children: [
                Icon(Icons.warning, color: Colors.white),
                SizedBox(width: 12),
                Text('Selecciona el nivel de gravedad del riesgo'),
              ],
            ),
            backgroundColor: Colors.orange,
          ),
        );
        return;
      }

      try {
        // 🆕 PRESERVAR HISTORIAL EXISTENTE
        final cultivoActualizado = Cultivo(
          id: widget.cultivo.id,
          nombre: widget.cultivo.nombre,
          tipoSuelo: widget.cultivo.tipoSuelo,
          area: widget.cultivo.area,
          fechaSiembra: widget.cultivo.fechaSiembra,
          fechaCosecha: widget.cultivo.fechaCosecha,
          estado: Cultivo.ESTADO_EN_RIESGO,
          notas: widget.cultivo.notas,
          imagenUrl: widget.cultivo.imagenUrl,
          tipoId: widget.cultivo.tipoId,
          categoriaId: widget.cultivo.categoriaId,
          tipoRiego: widget.cultivo.tipoRiego,
          cantidadCosechada: widget.cultivo.cantidadCosechada,
          ingresos: widget.cultivo.ingresos,
          egresos: widget.cultivo.egresos,
          razonRiesgo: _razonCtrl.text,
          tipoRiesgo: _tipoRiesgo,
          fechaRiesgo: DateTime.now().toIso8601String().split('T')[0],
          fechaInicioRiesgo: _fechaInicioRiesgo != null
              ? '${_fechaInicioRiesgo!.day.toString().padLeft(2, '0')}/${_fechaInicioRiesgo!.month.toString().padLeft(2, '0')}/${_fechaInicioRiesgo!.year}'
              : null,
          nivelGravedad: _nivelGravedad,
          // 🆕 PRESERVAR HISTORIAL ANTERIOR
          historialRiesgos: widget.cultivo.historialRiesgos,
        );

        // 🆕 DEBUG: Imprimir para verificar
        print(
          'RiesgoForm - Guardando nuevo riesgo, historial preservado: ${widget.cultivo.historialRiesgos}',
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
                  Icon(Icons.warning, color: Colors.white),
                  SizedBox(width: 12),
                  Text('Cultivo marcado en riesgo'),
                ],
              ),
              backgroundColor: Colors.orange,
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
        title: const Text('Registrar Riesgo'),
        backgroundColor: Colors.orange,
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
              // Información del cultivo
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.orange[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.orange, width: 2),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: const BoxDecoration(
                            color: Colors.orange,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.warning,
                            color: Colors.white,
                            size: 24,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Cultivo en riesgo',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.orange,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                widget.cultivo.nombre,
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Tipo de riesgo
              const Text(
                'Tipo de Riesgo',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _tiposRiesgo.map((tipo) {
                  final isSelected = _tipoRiesgo == tipo['value'];
                  return FilterChip(
                    selected: isSelected,
                    label: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          tipo['icon'],
                          size: 18,
                          color: isSelected ? Colors.white : tipo['color'],
                        ),
                        const SizedBox(width: 6),
                        Text(tipo['value']),
                      ],
                    ),
                    onSelected: (selected) {
                      setState(() {
                        _tipoRiesgo = selected ? tipo['value'] : null;
                      });
                    },
                    selectedColor: tipo['color'],
                    backgroundColor: (tipo['color'] as Color).withOpacity(0.1),
                    checkmarkColor: Colors.white,
                    labelStyle: TextStyle(
                      color: isSelected ? Colors.white : tipo['color'],
                      fontWeight: FontWeight.bold,
                    ),
                  );
                }).toList(),
              ),
              if (_tipoRiesgo != null) ...[
                const SizedBox(height: 8),
                Text(
                  _tiposRiesgo.firstWhere(
                    (t) => t['value'] == _tipoRiesgo,
                  )['description'],
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
              const SizedBox(height: 24),

              // Razón del riesgo
              TextFormField(
                controller: _razonCtrl,
                decoration: InputDecoration(
                  labelText: 'Razón Principal',
                  hintText: 'Ej: Ataque de pulgones en hojas',
                  prefixIcon: const Icon(
                    Icons.description,
                    color: Colors.orange,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.orange[50],
                ),
                maxLines: 2,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Describe la razón del riesgo';
                  }
                  if (value.length < 10) {
                    return 'Proporciona más detalles (mínimo 10 caracteres)';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Fecha de inicio del riesgo
              InkWell(
                onTap: () async {
                  final fecha = await showDatePicker(
                    context: context,
                    initialDate: _fechaInicioRiesgo ?? DateTime.now(),
                    firstDate: DateTime.now().subtract(
                      const Duration(days: 30),
                    ),
                    lastDate: DateTime.now().add(const Duration(days: 30)),
                  );
                  if (fecha != null) {
                    setState(() {
                      _fechaInicioRiesgo = fecha;
                    });
                  }
                },
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.orange),
                    borderRadius: BorderRadius.circular(12),
                    color: Colors.orange[50],
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.calendar_today, color: Colors.orange),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Fecha de Inicio del Riesgo',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.orange,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              _fechaInicioRiesgo != null
                                  ? '${_fechaInicioRiesgo!.day}/${_fechaInicioRiesgo!.month}/${_fechaInicioRiesgo!.year}'
                                  : 'Seleccionar fecha',
                              style: TextStyle(
                                fontSize: 16,
                                color: _fechaInicioRiesgo != null
                                    ? Colors.black
                                    : Colors.grey[600],
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const Icon(Icons.arrow_drop_down, color: Colors.orange),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Nivel de gravedad
              const Text(
                'Nivel de Gravedad',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _nivelesGravedad.map((nivel) {
                  final isSelected = _nivelGravedad == nivel['value'];
                  return FilterChip(
                    selected: isSelected,
                    label: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          nivel['icon'],
                          size: 18,
                          color: isSelected ? Colors.white : nivel['color'],
                        ),
                        const SizedBox(width: 6),
                        Text(nivel['value']),
                      ],
                    ),
                    onSelected: (selected) {
                      setState(() {
                        _nivelGravedad = selected ? nivel['value'] : null;
                      });
                    },
                    selectedColor: nivel['color'],
                    backgroundColor: (nivel['color'] as Color).withOpacity(0.1),
                    checkmarkColor: Colors.white,
                    labelStyle: TextStyle(
                      color: isSelected ? Colors.white : nivel['color'],
                      fontWeight: FontWeight.bold,
                    ),
                  );
                }).toList(),
              ),
              if (_nivelGravedad != null) ...[
                const SizedBox(height: 8),
                Text(
                  _nivelesGravedad.firstWhere(
                    (n) => n['value'] == _nivelGravedad,
                  )['description'],
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
              const SizedBox(height: 24),

              // Detalles adicionales
              TextFormField(
                controller: _detallesCtrl,
                decoration: InputDecoration(
                  labelText: 'Detalles Adicionales (Opcional)',
                  hintText: 'Acciones tomadas, observaciones...',
                  prefixIcon: const Icon(Icons.notes, color: Colors.orange),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.orange[50],
                ),
                maxLines: 4,
              ),
              const SizedBox(height: 24),

              // Información importante
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
                        'Este registro te ayudará a llevar un control de los problemas en tus cultivos y tomar mejores decisiones.',
                        style: TextStyle(fontSize: 12, color: Colors.blue[900]),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Botón guardar
              ElevatedButton.icon(
                onPressed: _tipoRiesgo == null ? null : _guardarRiesgo,
                icon: const Icon(Icons.save),
                label: const Text(
                  'Registrar Riesgo',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  disabledBackgroundColor: Colors.grey[300],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
