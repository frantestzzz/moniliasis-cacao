import 'package:flutter/material.dart';
import 'package:amgeca/classes/tipo_cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';

class TipoFormPage extends StatefulWidget {
  final TipoCultivo? tipo;
  const TipoFormPage({Key? key, this.tipo}) : super(key: key);

  @override
  State<TipoFormPage> createState() => _TipoFormPageState();
}

class _TipoFormPageState extends State<TipoFormPage> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _nombreCtrl;

  @override
  void initState() {
    super.initState();
    _nombreCtrl = TextEditingController(text: widget.tipo?.nombre ?? '');
  }

  @override
  void dispose() {
    _nombreCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    final row = {'nombre': _nombreCtrl.text};
    try {
      if (widget.tipo == null) {
        await BasedatoHelper.instance.insertTipoCultivo(row);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Tipo creado')));
      } else {
        // no update method for tipo; replace by delete+insert or implement update later
        await BasedatoHelper.instance.insertTipoCultivo({
          'id': widget.tipo!.id,
          'nombre': _nombreCtrl.text,
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Tipo actualizado')));
      }
      Navigator.of(context).pop(true);
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final isEditing = widget.tipo != null;
    return Scaffold(
      appBar: AppBar(title: Text(isEditing ? 'Editar Tipo' : 'Nuevo Tipo')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              TextFormField(
                controller: _nombreCtrl,
                decoration: const InputDecoration(
                  labelText: 'Nombre',
                  border: OutlineInputBorder(),
                ),
                validator: (v) => (v == null || v.isEmpty) ? 'Requerido' : null,
              ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: _save,
                icon: const Icon(Icons.save),
                label: const Text('Guardar'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
