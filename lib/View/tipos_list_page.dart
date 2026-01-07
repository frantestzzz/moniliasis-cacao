import 'package:flutter/material.dart';
import 'package:amgeca/classes/tipo_cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'tipo_form.dart';

class TiposListPage extends StatefulWidget {
  const TiposListPage({Key? key}) : super(key: key);

  @override
  State<TiposListPage> createState() => _TiposListPageState();
}

class _TiposListPageState extends State<TiposListPage> {
  late Future<List<TipoCultivo>> _future;

  @override
  void initState() {
    super.initState();
    _load();
  }

  void _load() {
    _future = _getAll();
  }

  Future<List<TipoCultivo>> _getAll() async {
    final rows = await BasedatoHelper.instance.getAllTiposCultivo();
    return rows.map((r) => TipoCultivo.fromMap(r)).toList();
  }

  Future<void> _delete(int id) async {
    await BasedatoHelper.instance.deleteTipoCultivo(id);
    setState(() => _load());
  }

  Future<void> _openForm({TipoCultivo? tipo}) async {
    final res = await Navigator.of(
      context,
    ).push(MaterialPageRoute(builder: (_) => TipoFormPage(tipo: tipo)));
    if (res == true) setState(() => _load());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tipos de cultivo')),
      body: FutureBuilder<List<TipoCultivo>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting)
            return const Center(child: CircularProgressIndicator());
          final items = snapshot.data ?? [];
          if (items.isEmpty)
            return Center(child: Text('No hay tipos. Agrega uno con +'));
          return ListView.separated(
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (context, i) {
              final t = items[i];
              return ListTile(
                title: Text(t.nombre),
                trailing: IconButton(
                  icon: const Icon(Icons.delete),
                  onPressed: () => _delete(t.id!),
                ),
                onTap: () => _openForm(tipo: t),
              );
            },
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _openForm,
        child: const Icon(Icons.add),
      ),
    );
  }
}
