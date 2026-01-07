import 'package:flutter/material.dart';
import 'package:amgeca/classes/categoria.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'categoria_form.dart';

class CategoriasListPage extends StatefulWidget {
  const CategoriasListPage({Key? key}) : super(key: key);

  @override
  State<CategoriasListPage> createState() => _CategoriasListPageState();
}

class _CategoriasListPageState extends State<CategoriasListPage> {
  late Future<List<Categoria>> _future;

  @override
  void initState() {
    super.initState();
    _load();
  }

  void _load() {
    _future = _getAll();
  }

  Future<List<Categoria>> _getAll() async {
    final rows = await BasedatoHelper.instance.getAllCategorias();
    return rows.map((r) => Categoria.fromMap(r)).toList();
  }

  Future<void> _delete(int id) async {
    await BasedatoHelper.instance.deleteCategoria(id);
    setState(() => _load());
  }

  Future<void> _openForm({Categoria? categoria}) async {
    final res = await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => CategoriaFormPage(categoria: categoria),
      ),
    );
    if (res == true) setState(() => _load());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Categorias')),
      body: FutureBuilder<List<Categoria>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting)
            return const Center(child: CircularProgressIndicator());
          final items = snapshot.data ?? [];
          if (items.isEmpty)
            return Center(child: Text('No hay categorías. Agrega una con +'));
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
                onTap: () => _openForm(categoria: t),
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
