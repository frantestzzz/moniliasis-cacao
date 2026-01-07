// lib/View/dashboard_page.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import 'package:amgeca/providers/auth_provider.dart';
import 'package:amgeca/providers/alerts_provider.dart';
import 'package:amgeca/providers/crops_provider.dart';
import 'package:amgeca/providers/tasks_provider.dart';
import 'package:amgeca/providers/stats_provider.dart';
import 'package:amgeca/classes/cronograma_actividad.dart';
import 'package:amgeca/classes/cultivo.dart';
import 'package:amgeca/Data/basedato_helper.dart';
import 'package:amgeca/View/ventas_page.dart';
import 'package:amgeca/View/ajustes_page.dart';
import 'package:amgeca/View/reportes.dart';
import 'package:amgeca/View/ayuda_page.dart';
import 'package:amgeca/View/acerca_de_page.dart';
import 'dart:io';
import 'package:flutter/painting.dart' show ImageProvider;

class DashboardPage extends StatefulWidget {
  const DashboardPage({Key? key}) : super(key: key);

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  bool _isInitialized = false;
  int _todayTasksCount = 0;
  int _cultivosEnRiesgoCount = 0;

  @override
  void initState() {
    super.initState();
    // Delay initialization to avoid calling setState during build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeData();
    });
  }

  Future<void> _initializeData() async {
    if (_isInitialized) return;

    try {
      final alertsProvider = Provider.of<AlertsProvider>(
        context,
        listen: false,
      );
      final tasksProvider = Provider.of<TasksProvider>(context, listen: false);
      final statsProvider = Provider.of<StatsProvider>(context, listen: false);
      final cropsProvider = Provider.of<CropsProvider>(context, listen: false);

      await Future.wait([
        cropsProvider.loadCultivos(),
        alertsProvider.loadAlerts(),
        tasksProvider.loadMockTasks(),
        statsProvider.loadStats(),
        _loadTodayTasksCount(),
        _loadCultivosEnRiesgoCount(),
      ]);

      setState(() => _isInitialized = true);
    } catch (e) {
      debugPrint('Error al inicializar datos: $e');
      setState(() => _isInitialized = true);
    }
  }

  Future<void> _loadTodayTasksCount() async {
    try {
      // Obtener fecha actual
      final now = DateTime.now();
      final today = DateTime(now.year, now.month, now.day);
      final todayStr = today.toIso8601String().split('T')[0];

      // Obtener todas las actividades pendientes de hoy
      final allCultivos = await BasedatoHelper.instance.getAllCultivos();
      int count = 0;

      for (final cultivoRow in allCultivos) {
        final cultivo = Cultivo.fromMap(cultivoRow);
        final actividades = await BasedatoHelper.instance
            .getCronogramaActividadesPorCultivo(cultivo.id!);

        for (final actividadRow in actividades) {
          final actividad = CronogramaActividad.fromMap(actividadRow);

          // Solo contar actividades pendientes o en progreso de hoy
          if ((actividad.estado == 'pendiente' ||
                  actividad.estado == 'en_progreso') &&
              actividad.fechaProgramada == todayStr) {
            count++;
          }
        }
      }

      setState(() {
        _todayTasksCount = count;
      });
    } catch (e) {
      debugPrint('Error al cargar conteo de tareas: $e');
      setState(() {
        _todayTasksCount = 0;
      });
    }
  }

  Future<void> _loadCultivosEnRiesgoCount() async {
    try {
      // Obtener todos los cultivos y contar los que están en riesgo
      final allCultivos = await BasedatoHelper.instance.getAllCultivos();
      int count = 0;

      for (final cultivoRow in allCultivos) {
        final cultivo = Cultivo.fromMap(cultivoRow);
        if (cultivo.esEnRiesgo) {
          count++;
        }
      }

      setState(() {
        _cultivosEnRiesgoCount = count;
      });
    } catch (e) {
      debugPrint('Error al cargar conteo de cultivos en riesgo: $e');
      setState(() {
        _cultivosEnRiesgoCount = 0;
      });
    }
  }

  Future<void> _refreshData() async {
    // Force refresh by resetting initialization flag
    setState(() => _isInitialized = false);
    await _initializeData();
  }

  @override
  Widget build(BuildContext context) {
    final authProvider = Provider.of<AuthProvider>(context);
    final cropsProvider = context.watch<CropsProvider>();
    final alertsProvider = context.watch<AlertsProvider>();
    final tasksProvider = context.watch<TasksProvider>();
    final statsProvider = context.watch<StatsProvider>();
    final userName = authProvider.user?['nombre'] ?? 'Usuario';

    if (!_isInitialized) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(color: Colors.green[700]),
              const SizedBox(height: 16),
              Text(
                'Cargando datos...',
                style: TextStyle(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Dashboard',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green[800],
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _refreshData,
            tooltip: 'Actualizar datos',
          ),
          IconButton(
            icon: const Icon(Icons.exit_to_app, color: Colors.white),
            onPressed: () => _handleLogout(context),
            tooltip: 'Cerrar sesión',
          ),
        ],
      ),
      drawer: _buildCustomDrawer(context, userName),
      body: RefreshIndicator(
        onRefresh: _refreshData,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: Column(
            children: [
              _buildWelcomePanel(userName),
              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Métricas de cultivos
                    _buildCropMetrics(cropsProvider, statsProvider),
                    const SizedBox(height: 20),

                    // Tarjetas de acción rápida
                    _buildQuickActions(
                      context,
                      alertsProvider,
                      tasksProvider,
                      cropsProvider,
                    ),
                    const SizedBox(height: 20),

                    // Próximas cosechas
                    _buildUpcomingHarvests(cropsProvider),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildWelcomePanel(String userName) {
    final now = DateTime.now();
    final greeting = now.hour < 12
        ? 'Buenos días'
        : now.hour < 18
        ? 'Buenas tardes'
        : 'Buenas noches';
    final formattedDate = DateFormat('dd/MM/yyyy').format(now);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.green[700]!, Colors.green[500]!],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            greeting.toUpperCase(),
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 13,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.5,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            userName.split(' ').first,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 32,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            formattedDate,
            style: const TextStyle(color: Colors.white70, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _buildCropMetrics(CropsProvider crops, StatsProvider stats) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '🌾 Estado de Cultivos',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _buildMetricCard(
                'Activos',
                '${crops.activeCount}',
                '${crops.totalArea.toStringAsFixed(1)} ha',
                Colors.green,
                Icons.check_circle,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildMetricCard(
                'En Riesgo',
                '${crops.enRiesgoCount}',
                'Requiere atención',
                Colors.orange,
                Icons.warning_amber,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _buildMetricCard(
                'Cosechados',
                '${crops.harvestedCount}',
                'Este mes',
                Colors.blue,
                Icons.emoji_events,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildMetricCard(
                'Inactivos',
                '${crops.inactiveCount}',
                'Sin actividad',
                Colors.grey,
                Icons.cancel,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildMetricCard(
    String title,
    String value,
    String detail,
    Color color,
    IconData icon,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: LinearGradient(
          colors: [color.withOpacity(0.9), color],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Icon(icon, color: Colors.white70, size: 20),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            detail,
            style: const TextStyle(color: Colors.white70, fontSize: 11),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickActions(
    BuildContext context,
    AlertsProvider alerts,
    TasksProvider tasks,
    CropsProvider crops,
  ) {
    final actions = [
      {
        'title': 'Alertas críticas',
        'value': '$_cultivosEnRiesgoCount',
        'icon': Icons.warning,
        'color': Colors.red,
        'route': '/alerts',
      },
      {
        'title': 'Tareas hoy',
        'value': '$_todayTasksCount',
        'icon': Icons.today,
        'color': Colors.blue,
        'route': '/tasks',
      },
      {
        'title': 'Ventas',
        'value': 'Ver',
        'icon': Icons.shopping_cart,
        'color': Colors.purple,
        'route': '/ventas',
      },
      {
        'title': 'Reportes',
        'value': 'Ver',
        'icon': Icons.analytics,
        'color': Colors.teal,
        'route': null,
      },
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '⚡ Acciones Rápidas',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: actions
              .map((action) => _buildActionCard(context, action))
              .toList(),
        ),
      ],
    );
  }

  Widget _buildActionCard(BuildContext context, Map<String, dynamic> action) {
    final width = (MediaQuery.of(context).size.width - 48) / 2;
    return GestureDetector(
      onTap: () {
        if (action['route'] != null) {
          Navigator.pushNamed(context, action['route']);
        } else {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const ReportesScreen()),
          );
        }
      },
      child: Container(
        width: width,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: action['color'].withOpacity(0.3)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: action['color'].withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(action['icon'], color: action['color'], size: 24),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    action['title'],
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    action['value'],
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: action['color'],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUpcomingHarvests(CropsProvider crops) {
    // Obtener todos los cultivos y filtrar solo activos y en riesgo
    final allCultivos = crops.cultivos;
    final filteredCultivos =
        allCultivos
            .where((cultivo) {
              // Solo mostrar cultivos que estén activos o en riesgo
              return cultivo.estado == 'activo' || cultivo.esEnRiesgo;
            })
            .where((cultivo) {
              // Solo mostrar cultivos con fecha de cosecha futura
              if (cultivo.fechaCosecha == null) return false;

              try {
                final fechaCosecha = DateTime.parse(cultivo.fechaCosecha!);
                return fechaCosecha.isAfter(DateTime.now());
              } catch (e) {
                return false;
              }
            })
            .toList()
          ..sort((a, b) {
            // Ordenar por fecha de cosecha (más cercana primero)
            if (a.fechaCosecha == null) return 1;
            if (b.fechaCosecha == null) return -1;

            try {
              final fechaA = DateTime.parse(a.fechaCosecha!);
              final fechaB = DateTime.parse(b.fechaCosecha!);
              return fechaA.compareTo(fechaB);
            } catch (e) {
              return 0;
            }
          })
          ..take(5); // Limitar a 5 cultivos

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '📅 Próximas Cosechas',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        if (filteredCultivos.isEmpty)
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey[100],
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Center(
              child: Column(
                children: [
                  Icon(Icons.calendar_today, size: 48, color: Colors.grey),
                  SizedBox(height: 8),
                  Text(
                    'No hay cosechas programadas',
                    style: TextStyle(color: Colors.grey),
                  ),
                  SizedBox(height: 4),
                  Text(
                    'Solo se muestran cultivos activos y en riesgo',
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ),
          )
        else
          ...filteredCultivos.map((cultivo) {
            final fechaCosecha = DateTime.tryParse(cultivo.fechaCosecha ?? '');
            final diasRestantes =
                fechaCosecha?.difference(DateTime.now()).inDays ?? 0;

            // Determinar colores según estado
            Color cardColor = Colors.white;
            Color borderColor = Colors.green.withOpacity(0.3);
            Color textColor = Colors.green;
            String estadoIcon = '🌱';

            if (cultivo.esEnRiesgo) {
              cardColor = Colors.red[50]!;
              borderColor = Colors.red.withOpacity(0.3);
              textColor = Colors.red;
              estadoIcon = '⚠️';
            } else if (diasRestantes <= 7) {
              cardColor = Colors.orange[50]!;
              borderColor = Colors.orange.withOpacity(0.3);
              textColor = Colors.orange;
              estadoIcon = '⚡';
            }

            return Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: borderColor),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: cardColor,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      estadoIcon,
                      style: const TextStyle(fontSize: 20),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              cultivo.nombre,
                              style: const TextStyle(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            if (cultivo.esEnRiesgo) ...[
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 6,
                                  vertical: 2,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.red,
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: const Text(
                                  'RIESGO',
                                  style: TextStyle(
                                    fontSize: 8,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                        Text(
                          '${cultivo.area} ha • ${cultivo.tipoSuelo}',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                        if (cultivo.esEnRiesgo &&
                            cultivo.notas != null &&
                            cultivo.notas!.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.only(top: 2),
                            child: Text(
                              cultivo.notas!,
                              style: TextStyle(
                                fontSize: 10,
                                color: Colors.red[600],
                                fontStyle: FontStyle.italic,
                              ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                      ],
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        '$diasRestantes días',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: textColor,
                        ),
                      ),
                      Text(
                        DateFormat('dd/MM').format(fechaCosecha!),
                        style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                      ),
                    ],
                  ),
                ],
              ),
            );
          }).toList(),
      ],
    );
  }

  Widget _buildCustomDrawer(BuildContext context, String userName) {
    return Drawer(
      width: MediaQuery.of(context).size.width * 0.65,
      child: Container(
        color: Colors.white,
        child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(20, 50, 20, 20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.green[700]!, Colors.green[500]!],
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  FutureBuilder<Map<String, dynamic>?>(
                    future: BasedatoHelper.instance.getUsuario(
                      Provider.of<AuthProvider>(
                            context,
                            listen: false,
                          ).user?['id'] ??
                          1,
                    ),
                    builder: (context, snapshot) {
                      String? imagenPath = snapshot.data?['imagenPerfil'];

                      return GestureDetector(
                        onTap: () {
                          if (imagenPath != null && imagenPath.isNotEmpty) {
                            Navigator.push(
                              context,
                              MaterialPageRoute(builder: (_) => AjustesPage()),
                            );
                          }
                        },
                        child: CircleAvatar(
                          radius: 35,
                          backgroundColor: Colors.white,
                          backgroundImage:
                              (imagenPath != null && imagenPath.isNotEmpty)
                              ? FileImage(File(imagenPath)) as ImageProvider
                              : null,
                          child: (imagenPath == null || imagenPath.isEmpty)
                              ? const Icon(
                                  Icons.person,
                                  size: 40,
                                  color: Colors.green,
                                )
                              : null,
                        ),
                      );
                    },
                  ),
                  const SizedBox(height: 12),
                  Text(
                    userName,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Agricultor',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.9),
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                padding: EdgeInsets.zero,
                children: [
                  const SizedBox(height: 8),
                  _buildDrawerItem(
                    context,
                    icon: Icons.analytics,
                    title: 'Reportes',
                    subtitle: 'Estadísticas',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const ReportesScreen(),
                        ),
                      );
                    },
                  ),
                  _buildDrawerItem(
                    context,
                    icon: Icons.attach_money,
                    title: 'Ventas',
                    subtitle: 'Gestión de ventas',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const VentasPage()),
                      );
                    },
                  ),
                  const Divider(height: 1),
                  _buildDrawerItem(
                    context,
                    icon: Icons.settings,
                    title: 'Ajustes',
                    subtitle: 'Configuración',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const AjustesPage()),
                      );
                    },
                  ),
                  const Divider(height: 1),
                  _buildDrawerItem(
                    context,
                    icon: Icons.help_outline,
                    title: 'Ayuda',
                    subtitle: 'Manual de usuario',
                    onTap: () {
                      _mostrarAyuda();
                    },
                  ),
                  const Divider(height: 1),
                  _buildDrawerItem(
                    context,
                    icon: Icons.info_outline,
                    title: 'Acerca de',
                    subtitle: 'Información de la aplicación',
                    onTap: () {
                      _mostrarAcercaDe();
                    },
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  const Divider(height: 1),
                  const SizedBox(height: 12),
                  Text(
                    'AMGeCCA v1.0.0',
                    style: TextStyle(color: Colors.grey[600], fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDrawerItem(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.green[50],
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(icon, color: Colors.green[700], size: 24),
      ),
      title: Text(
        title,
        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        subtitle,
        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
      ),
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
    );
  }

  Future<void> _handleLogout(BuildContext context) async {
    final shouldLogout = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Cerrar sesión'),
        content: const Text('¿Estás seguro?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancelar'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text(
              'Cerrar sesión',
              style: TextStyle(color: Colors.red),
            ),
          ),
        ],
      ),
    );

    if (shouldLogout == true && context.mounted) {
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      await authProvider.logout();
      if (context.mounted)
        Navigator.of(
          context,
        ).pushNamedAndRemoveUntil('/login', (route) => false);
    }
  }

  void _mostrarAyuda() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const AyudaPage()),
    );
  }

  void _mostrarAcercaDe() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const AcercaDePage()),
    );
  }
}
