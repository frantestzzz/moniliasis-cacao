// lib/main.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:amgeca/providers/auth_provider.dart';
import 'package:amgeca/providers/configuracion_provider.dart';
import 'package:amgeca/providers/crops_provider.dart';
import 'package:amgeca/providers/alerts_provider.dart';
import 'package:amgeca/providers/tasks_provider.dart';
import 'package:amgeca/providers/stats_provider.dart';
import 'package:amgeca/providers/weather_provider.dart';
import 'package:amgeca/View/ventas_page.dart';
import 'package:amgeca/View/auth/login_page.dart';
import 'package:amgeca/View/alerts_page.dart';
import 'package:amgeca/View/tasks_page.dart';
import 'package:amgeca/View/riego_page.dart';
import 'package:amgeca/View/cosecha_page.dart';
import 'package:amgeca/View/clima_page.dart';
import 'package:amgeca/View/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(
          create: (_) => ConfiguracionProvider()..cargarConfiguracion(),
        ),
        ChangeNotifierProvider(create: (_) => CropsProvider()),
        ChangeNotifierProvider(create: (_) => AlertsProvider()),
        ChangeNotifierProvider(create: (_) => TasksProvider()),
        ChangeNotifierProvider(create: (_) => StatsProvider()),
        ChangeNotifierProvider(create: (_) => WeatherProvider()),
      ],
      child: Consumer<ConfiguracionProvider>(
        builder: (context, configProvider, child) {
          return MaterialApp(
            title: 'AMGECA',
            theme: ThemeData(
              primarySwatch: Colors.green,
              visualDensity: VisualDensity.adaptivePlatformDensity,
              useMaterial3: true,
              colorScheme: ColorScheme.fromSeed(
                seedColor: Colors.green,
                brightness: Brightness.light,
              ),
            ),
            darkTheme: ThemeData(
              primarySwatch: Colors.green,
              visualDensity: VisualDensity.adaptivePlatformDensity,
              useMaterial3: true,
              colorScheme: ColorScheme.fromSeed(
                seedColor: Colors.green,
                brightness: Brightness.dark,
              ),
            ),
            themeMode: configProvider.tema == 'dark'
                ? ThemeMode.dark
                : ThemeMode.light,
            home: const SplashScreen(),
            debugShowCheckedModeBanner: false,
            routes: {
              '/splash': (context) => const SplashScreen(),
              '/ventas': (context) => const VentasPage(),
              '/inventario': (context) => const InventarioPage(),
              '/reportes-ventas': (context) =>
                  const ReportesVentasPlaceholder(),
              '/login': (context) => const LoginPage(),
              '/alerts': (context) => const AlertsPage(),
              '/tasks': (context) => const TasksPage(),
              '/riego': (context) => const RiegoPage(),
              '/cosecha': (context) => const CosechaPage(),
              '/clima': (context) => const ClimaPage(),
            },
          );
        },
      ),
    );
  }
}

class ReportesVentasPlaceholder extends StatelessWidget {
  const ReportesVentasPlaceholder({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Reportes de Ventas',
          style: TextStyle(color: Colors.white),
        ),
        backgroundColor: Colors.green[800],
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.analytics, size: 80, color: Colors.grey),
            SizedBox(height: 16),
            Text(
              'Reportes de Ventas',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text('Esta sección todavía está en desarrollo...'),
          ],
        ),
      ),
    );
  }
}

// Mock inventory provider
class InventoryItem {
  final String id;
  final String name;
  final String category;
  final double stock;
  final String unit;
  final DateTime lastUpdated;

  InventoryItem({
    required this.id,
    required this.name,
    required this.category,
    required this.stock,
    required this.unit,
    required this.lastUpdated,
  });
}

class InventoryProvider with ChangeNotifier {
  final List<InventoryItem> _items = [];
  List<InventoryItem> get items => List.unmodifiable(_items);

  void loadMockItems() {
    _items
      ..clear()
      ..addAll([
        InventoryItem(
          id: 'inv1',
          name: 'Fertilizante NPK 15-15-15',
          category: 'Fertilizantes',
          stock: 120.5,
          unit: 'kg',
          lastUpdated: DateTime.now().subtract(const Duration(days: 2)),
        ),
        InventoryItem(
          id: 'inv2',
          name: 'Semillas de Tomate',
          category: 'Semillas',
          stock: 8,
          unit: 'bolsas',
          lastUpdated: DateTime.now().subtract(const Duration(days: 5)),
        ),
        InventoryItem(
          id: 'inv3',
          name: 'Insecticida Agrícola',
          category: 'Plaguicidas',
          stock: 15.2,
          unit: 'L',
          lastUpdated: DateTime.now().subtract(const Duration(days: 1)),
        ),
      ]);
    notifyListeners();
  }
}

class InventarioPage extends StatelessWidget {
  const InventarioPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Inventario', style: TextStyle(color: Colors.white)),
        backgroundColor: Colors.green[800],
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: ChangeNotifierProvider(
        create: (_) => InventoryProvider()..loadMockItems(),
        child: Consumer<InventoryProvider>(
          builder: (context, provider, _) {
            final items = provider.items;
            if (items.isEmpty) {
              return const Center(
                child: Text('No hay ítems en el inventario.'),
              );
            }
            return ListView.separated(
              padding: const EdgeInsets.all(16),
              separatorBuilder: (_, __) => const Divider(),
              itemCount: items.length,
              itemBuilder: (context, index) {
                final item = items[index];
                final isLow = item.stock < 20;
                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: isLow
                        ? Colors.red[100]
                        : Colors.green[100],
                    child: Icon(
                      Icons.inventory,
                      color: isLow ? Colors.red : Colors.green,
                    ),
                  ),
                  title: Text(item.name),
                  subtitle: Text(
                    '${item.category} • Actualizado: ${_formatDate(item.lastUpdated)}',
                  ),
                  trailing: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        '${item.stock} ${item.unit}',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: isLow ? Colors.red : Colors.black,
                        ),
                      ),
                      if (isLow)
                        const Text(
                          'Stock bajo',
                          style: TextStyle(fontSize: 11, color: Colors.red),
                        ),
                    ],
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day.toString().padLeft(2, '0')}/${date.month.toString().padLeft(2, '0')}/${date.year}';
  }
}
