// lib/View/cultivo_form.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/classes/tipo_cultivo.dart';
import 'package:amgeca/classes/categoria.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:amgeca/services/image_service.dart';

class CultivoFormPage extends StatefulWidget {
  final Cultivo? cultivo;

  const CultivoFormPage({Key? key, this.cultivo}) : super(key: key);

  @override
  State<CultivoFormPage> createState() => _CultivoFormPageState();
}

class _CultivoFormPageState extends State<CultivoFormPage> {
  late TextEditingController _nombreCtrl;
  String? _selectedTipoSuelo;
  late TextEditingController _areaCtrl;
  late TextEditingController _fechaSiembraCtrl;
  late TextEditingController _fechaCosechaCtrl;
  String _selectedEstado = Cultivo.ESTADO_ACTIVO;
  late TextEditingController _notasCtrl;
  String? _selectedTipoRiego;
  String? _imagenPath;
  bool _imagenCambiada = false;

  final ImageService _imageService = ImageService();

  final List<String> _tipoSueloOptions = [
    'Arenoso',
    'Arcilloso',
    'Franco',
    'Limoso',
    'Humífero',
    'Pedregoso',
  ];

  final List<String> _tipoRiegoOptions = [
    'Goteo',
    'Aspersión',
    'Inundación',
    'Presurizado',
    'Lluvia Natural',
  ];

  final Map<String, Map<String, dynamic>> _estadosInfo = {
    Cultivo.ESTADO_ACTIVO: {
      'label': 'Activo',
      'icon': Icons.check_circle,
      'color': Colors.green,
    },
    Cultivo.ESTADO_EN_RIESGO: {
      'label': 'En Riesgo',
      'icon': Icons.warning,
      'color': Colors.orange,
    },
    Cultivo.ESTADO_COSECHADO: {
      'label': 'Cosechado',
      'icon': Icons.agriculture,
      'color': Colors.amber,
    },
    Cultivo.ESTADO_INACTIVO: {
      'label': 'Inactivo',
      'icon': Icons.cancel,
      'color': Colors.grey,
    },
  };

  List<TipoCultivo> _tipos = [];
  List<Categoria> _categorias = [];
  int? _selectedTipoId;
  int? _selectedCategoriaId;

  final _formKey = GlobalKey<FormState>();

  @override
  void initState() {
    super.initState();
    _nombreCtrl = TextEditingController(text: widget.cultivo?.nombre ?? '');
    _selectedTipoSuelo = widget.cultivo?.tipoSuelo ?? _tipoSueloOptions.first;
    _areaCtrl = TextEditingController(
      text: widget.cultivo?.area.toStringAsFixed(2) ?? '',
    );
    _fechaSiembraCtrl = TextEditingController(
      text: widget.cultivo?.fechaSiembra ?? '',
    );
    _fechaCosechaCtrl = TextEditingController(
      text: widget.cultivo?.fechaCosecha ?? '',
    );
    _selectedEstado = widget.cultivo?.estado ?? Cultivo.ESTADO_ACTIVO;
    _notasCtrl = TextEditingController(text: widget.cultivo?.notas ?? '');
    _selectedTipoRiego = widget.cultivo?.tipoRiego ?? _tipoRiegoOptions.first;
    _imagenPath = widget.cultivo?.imagenUrl;
    _selectedTipoId = widget.cultivo?.tipoId;
    _selectedCategoriaId = widget.cultivo?.categoriaId;
    _loadTiposYCategorias();
  }

  Future<void> _loadTiposYCategorias() async {
    final tiposRows = await BasedatoHelper.instance.getAllTiposCultivo();
    final categoriasRows = await BasedatoHelper.instance.getAllCategorias();
    setState(() {
      _tipos = tiposRows.map((r) => TipoCultivo.fromMap(r)).toList();
      _categorias = categoriasRows.map((r) => Categoria.fromMap(r)).toList();
    });
  }

  @override
  void dispose() {
    _nombreCtrl.dispose();
    _areaCtrl.dispose();
    _fechaSiembraCtrl.dispose();
    _fechaCosechaCtrl.dispose();
    _notasCtrl.dispose();
    super.dispose();
  }

  Future<void> _selectDate(TextEditingController controller) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime.now(), // No permitir fechas pasadas
      lastDate: DateTime(2100),
    );
    if (picked != null) {
      controller.text = picked.toIso8601String().split('T')[0];
      // Validar fechas después de seleccionar
      _validateFechas();
    }
  }

  void _validateFechas() {
    // Forzar validación cruzada de fechas
    if (_formKey.currentState != null) {
      _formKey.currentState!.validate();
    }
  }

  Future<void> _mostrarOpcionesImagen() async {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Seleccionar imagen',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.blue[50],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.camera_alt, color: Colors.blue),
                ),
                title: const Text('Tomar foto'),
                onTap: () async {
                  Navigator.pop(context);
                  await _capturarImagen(ImageSource.camera);
                },
              ),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.green[50],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.photo_library, color: Colors.green),
                ),
                title: const Text('Seleccionar de galería'),
                onTap: () async {
                  Navigator.pop(context);
                  await _capturarImagen(ImageSource.gallery);
                },
              ),
              if (_imagenPath != null)
                ListTile(
                  leading: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.red[50],
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.delete, color: Colors.red),
                  ),
                  title: const Text('Eliminar imagen'),
                  onTap: () {
                    Navigator.pop(context);
                    setState(() {
                      _imagenPath = null;
                      _imagenCambiada = true;
                    });
                  },
                ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _capturarImagen(ImageSource source) async {
    final String? newImagePath = await _imageService.seleccionarImagen(
      source: source,
    );

    if (newImagePath != null) {
      // Si había una imagen anterior y se cambió, eliminarla
      if (_imagenPath != null && _imagenCambiada) {
        await _imageService.eliminarImagen(_imagenPath);
      }

      setState(() {
        _imagenPath = newImagePath;
        _imagenCambiada = true;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('✅ Imagen agregada exitosamente'),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  Future<void> _saveCultivo() async {
    if (_formKey.currentState!.validate()) {
      try {
        final cultivo = Cultivo(
          id: widget.cultivo?.id,
          nombre: _nombreCtrl.text,
          tipoSuelo: _selectedTipoSuelo ?? '',
          area: double.parse(_areaCtrl.text),
          fechaSiembra: _fechaSiembraCtrl.text,
          fechaCosecha: _fechaCosechaCtrl.text.isEmpty
              ? null
              : _fechaCosechaCtrl.text,
          estado: _selectedEstado,
          notas: _notasCtrl.text.isEmpty ? null : _notasCtrl.text,
          imagenUrl: _imagenPath,
          tipoId: _selectedTipoId,
          categoriaId: _selectedCategoriaId,
          tipoRiego: _selectedTipoRiego ?? '',
        );

        if (widget.cultivo == null) {
          await BasedatoHelper.instance.insertCultivo(cultivo.toMap());
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Cultivo creado exitosamente')),
          );
        } else {
          await BasedatoHelper.instance.updateCultivo(
            widget.cultivo!.id!,
            cultivo.toMap(),
          );
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Cultivo actualizado exitosamente')),
          );
        }

        Navigator.of(context).pop(true);
      } catch (e) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('❌ Error: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isEditing = widget.cultivo != null;
    return Scaffold(
      appBar: AppBar(
        title: Text(isEditing ? 'Editar Cultivo' : 'Nuevo Cultivo'),
        backgroundColor: Colors.green,
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
              // Sección de imagen
              _buildImagenSection(),
              const SizedBox(height: 24),

              // Nombre del cultivo
              TextFormField(
                controller: _nombreCtrl,
                decoration: InputDecoration(
                  labelText: 'Nombre del Cultivo',
                  prefixIcon: const Icon(
                    Icons.agriculture,
                    color: Colors.green,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.green[50],
                ),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(
                    RegExp(r'^[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+$'),
                  ),
                ],
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'El nombre del cultivo es requerido';
                  }
                  if (value.length < 3) {
                    return 'El nombre debe tener al menos 3 caracteres';
                  }
                  if (!RegExp(r'^[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+$').hasMatch(value)) {
                    return 'Solo se permiten letras, números, tildes y ñ';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Tipo de cultivo y categoría
              Row(
                children: [
                  Expanded(child: _buildTipoDropdown()),
                  const SizedBox(width: 12),
                  Expanded(child: _buildCategoriaDropdown()),
                ],
              ),
              const SizedBox(height: 16),

              // Tipo de riego y suelo
              Row(
                children: [
                  Expanded(child: _buildTipoRiegoDropdown()),
                  const SizedBox(width: 12),
                  Expanded(child: _buildTipoSueloDropdown()),
                ],
              ),
              const SizedBox(height: 16),

              // Área
              TextFormField(
                controller: _areaCtrl,
                decoration: InputDecoration(
                  labelText: 'Área (ha)',
                  prefixIcon: const Icon(Icons.straighten, color: Colors.green),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.green[50],
                ),
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'^\d*\.?\d*')),
                ],
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'El área en hectáreas es requerida';
                  }

                  // Validar formato numérico
                  if (!RegExp(r'^\d*\.?\d*$').hasMatch(value)) {
                    return 'Solo se permiten números';
                  }

                  final area = double.tryParse(value);
                  if (area == null) {
                    return 'Ingresa un número válido';
                  }

                  if (area <= 0) {
                    return 'El área debe ser mayor a 0';
                  }

                  if (area > 100000) {
                    return 'El área no puede ser mayor a 10,000 ha';
                  }

                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Fechas
              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: _fechaSiembraCtrl,
                      decoration: InputDecoration(
                        labelText: 'Fecha de Siembra',
                        prefixIcon: const Icon(
                          Icons.calendar_today,
                          color: Colors.green,
                        ),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        filled: true,
                        fillColor: Colors.green[50],
                      ),
                      readOnly: true,
                      onTap: () => _selectDate(_fechaSiembraCtrl),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Fecha requerida';
                        }

                        // Validar que no sea fecha pasada
                        final selectedDate = DateTime.tryParse(value);
                        if (selectedDate != null) {
                          final now = DateTime.now();
                          final today = DateTime(now.year, now.month, now.day);
                          final selectedDay = DateTime(
                            selectedDate.year,
                            selectedDate.month,
                            selectedDate.day,
                          );

                          if (selectedDay.isBefore(today)) {
                            return 'No se pueden seleccionar fechas pasadas';
                          }
                        }

                        // Validar que sea anterior a la fecha de cosecha
                        if (_fechaCosechaCtrl.text.isNotEmpty) {
                          final fechaCosecha = DateTime.tryParse(
                            _fechaCosechaCtrl.text,
                          );
                          if (fechaCosecha != null && selectedDate != null) {
                            if (selectedDate.isAfter(fechaCosecha)) {
                              return 'Fecha anterior a cosecha';
                            }
                          }
                        }

                        return null;
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: TextFormField(
                      controller: _fechaCosechaCtrl,
                      decoration: InputDecoration(
                        labelText: 'Cosecha estimada',
                        prefixIcon: const Icon(
                          Icons.event_available,
                          color: Colors.green,
                        ),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        filled: true,
                        fillColor: Colors.green[50],
                      ),
                      readOnly: true,
                      onTap: () => _selectDate(_fechaCosechaCtrl),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Fecha requerida';
                        }

                        // Validar que no sea fecha pasada
                        final selectedDate = DateTime.tryParse(value);
                        if (selectedDate != null) {
                          final now = DateTime.now();
                          final today = DateTime(now.year, now.month, now.day);
                          final selectedDay = DateTime(
                            selectedDate.year,
                            selectedDate.month,
                            selectedDate.day,
                          );

                          if (selectedDay.isBefore(today)) {
                            return 'No se pueden seleccionar fechas pasadas';
                          }
                        }

                        // Validar que sea posterior a la fecha de siembra
                        if (_fechaSiembraCtrl.text.isNotEmpty) {
                          final fechaSiembra = DateTime.tryParse(
                            _fechaSiembraCtrl.text,
                          );
                          if (fechaSiembra != null && selectedDate != null) {
                            if (selectedDate.isBefore(fechaSiembra)) {
                              return 'Fecha posterior a siembra';
                            }
                          }
                        }

                        return null;
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Estado del cultivo
              _buildEstadoSelector(),
              const SizedBox(height: 16),

              // Notas
              TextFormField(
                controller: _notasCtrl,
                decoration: InputDecoration(
                  labelText: 'Notas (Opcional)',
                  prefixIcon: const Icon(Icons.notes, color: Colors.green),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.green[50],
                ),
                maxLines: 3,
              ),
              const SizedBox(height: 24),

              // Botón guardar
              ElevatedButton.icon(
                onPressed: _saveCultivo,
                icon: const Icon(Icons.save),
                label: Text(
                  isEditing ? 'Actualizar Cultivo' : 'Crear Cultivo',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagenSection() {
    return GestureDetector(
      onTap: _mostrarOpcionesImagen,
      child: Container(
        height: 200,
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.green, width: 2),
        ),
        child: _imagenPath != null && File(_imagenPath!).existsSync()
            ? ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.file(File(_imagenPath!), fit: BoxFit.cover),
                    Positioned(
                      top: 8,
                      right: 8,
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Icon(
                          Icons.edit,
                          color: Colors.white,
                          size: 20,
                        ),
                      ),
                    ),
                  ],
                ),
              )
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.add_a_photo, size: 48, color: Colors.green[700]),
                  const SizedBox(height: 8),
                  Text(
                    'Toca para agregar foto',
                    style: TextStyle(
                      color: Colors.green[700],
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Cámara o Galería',
                    style: TextStyle(color: Colors.grey[600], fontSize: 12),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildTipoDropdown() {
    return DropdownButtonFormField<int>(
      value: _selectedTipoId,
      isExpanded: true,
      decoration: InputDecoration(
        labelText: 'Tipo',
        prefixIcon: const Icon(Icons.category, color: Colors.green),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        filled: true,
        fillColor: Colors.green[50],
      ),
      items: _tipos
          .map((t) => DropdownMenuItem<int>(value: t.id, child: Text(t.nombre)))
          .toList(),
      onChanged: (v) => setState(() => _selectedTipoId = v),
      validator: (value) {
        if (value == null) return 'Requerido';
        return null;
      },
    );
  }

  Widget _buildCategoriaDropdown() {
    return DropdownButtonFormField<int>(
      value: _selectedCategoriaId,
      isExpanded: true,
      decoration: InputDecoration(
        labelText: 'Categoría',
        prefixIcon: const Icon(Icons.label, color: Colors.green),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        filled: true,
        fillColor: Colors.green[50],
      ),
      items: _categorias
          .map((c) => DropdownMenuItem<int>(value: c.id, child: Text(c.nombre)))
          .toList(),
      onChanged: (v) => setState(() => _selectedCategoriaId = v),
      validator: (value) {
        if (value == null) return 'Requerido';
        return null;
      },
    );
  }

  Widget _buildTipoRiegoDropdown() {
    return DropdownButtonFormField<String>(
      value: _selectedTipoRiego,
      isExpanded: true,
      decoration: InputDecoration(
        labelText: 'Riego',
        prefixIcon: const Icon(Icons.water, color: Colors.green),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        filled: true,
        fillColor: Colors.green[50],
      ),
      items: _tipoRiegoOptions
          .map((r) => DropdownMenuItem(value: r, child: Text(r)))
          .toList(),
      onChanged: (v) => setState(() => _selectedTipoRiego = v),
      validator: (v) {
        if (v == null || v.isEmpty) return 'Requerido';
        return null;
      },
    );
  }

  Widget _buildTipoSueloDropdown() {
    return DropdownButtonFormField<String>(
      value: _selectedTipoSuelo,
      isExpanded: true,
      decoration: InputDecoration(
        labelText: 'Suelo',
        prefixIcon: const Icon(Icons.landscape, color: Colors.green),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        filled: true,
        fillColor: Colors.green[50],
      ),
      items: _tipoSueloOptions
          .map((s) => DropdownMenuItem(value: s, child: Text(s)))
          .toList(),
      onChanged: (v) => setState(() => _selectedTipoSuelo = v),
      validator: (value) {
        if (value == null || value.isEmpty) return 'Requerido';
        return null;
      },
    );
  }

  Widget _buildEstadoSelector() {
    final estadosDisponibles = widget.cultivo != null
        ? _estadosInfo.entries
        : _estadosInfo.entries.where(
            (entry) => entry.key == Cultivo.ESTADO_ACTIVO,
          );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Estado del Cultivo',
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: estadosDisponibles.map((entry) {
            final estado = entry.key;
            final info = entry.value;
            final isSelected = _selectedEstado == estado;

            return FilterChip(
              selected: isSelected,
              label: Text(info['label']),
              avatar: Icon(
                info['icon'],
                size: 18,
                color: isSelected ? Colors.white : info['color'],
              ),
              onSelected: (selected) {
                if (selected) {
                  setState(() => _selectedEstado = estado);
                }
              },
              selectedColor: info['color'],
              backgroundColor: (info['color'] as Color).withOpacity(0.1),
              checkmarkColor: Colors.white,
              labelStyle: TextStyle(
                color: isSelected ? Colors.white : info['color'],
                fontWeight: FontWeight.bold,
              ),
            );
          }).toList(),
        ),
      ],
    );
  }
}
