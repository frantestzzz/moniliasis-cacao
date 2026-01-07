import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:provider/provider.dart';
import '../Data/basedato_helper.dart';
import '../providers/stats_provider.dart';

class ReportesScreen extends StatefulWidget {
  const ReportesScreen({super.key});

  @override
  State<ReportesScreen> createState() => _ReportesScreenState();
}

class _ReportesScreenState extends State<ReportesScreen> {
  List<Map<String, Object?>> cultivos = [];
  List<Map<String, Object?>> ventas = [];
  List<Map<String, Object?>> categorias = [];
  List<Map<String, Object?>> tipos = [];

  bool loading = true;

  // Función para generar color consistente basado en ID
  Color getColorFromId(String id, {int shade = 300}) {
    final colors = [
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.purple,
      Colors.red,
      Colors.teal,
      Colors.indigo,
      Colors.pink,
      Colors.amber,
      Colors.cyan,
      Colors.lime,
      Colors.brown,
      Colors.grey,
      Colors.blueGrey,
      Colors.deepOrange,
      Colors.deepPurple,
      Colors.lightBlue,
      Colors.lightGreen,
      Colors.yellow,
      Colors.cyan,
    ];

    final colorIndex = int.tryParse(id) ?? id.hashCode.abs();
    final baseColor = colors[colorIndex % colors.length];

    // Usar diferentes shades para diferentes gráficos
    switch (shade) {
      case 200:
        return baseColor.shade200;
      case 300:
        return baseColor.shade300;
      case 400:
        return baseColor.shade400;
      default:
        return baseColor.shade300;
    }
  }

  @override
  void initState() {
    super.initState();
    cargarDatos();
  }

  Future<void> cargarDatos() async {
    final db = BasedatoHelper.instance;

    final c = await db.getAllCultivos();
    final v = await db.getAllVentas();
    final cat = await db.getAllCategorias();
    final tip = await db.getAllTiposCultivo();

    setState(() {
      cultivos = c;
      ventas = v;
      categorias = cat;
      tipos = tip;
      loading = false;
    });
  }

  // ============================================================
  // 1. Ventas por mes
  // ============================================================
  List<BarChartGroupData> getVentasPorMes() {
    final Map<int, double> meses = {for (var i = 1; i <= 12; i++) i: 0.0};

    for (var venta in ventas) {
      final fecha = DateTime.tryParse(venta["fecha"].toString());
      if (fecha != null) {
        meses[fecha.month] =
            meses[fecha.month]! + (venta["total"] as num).toDouble();
      }
    }

    return meses.entries.map((e) {
      return BarChartGroupData(
        x: e.key,
        barRods: [BarChartRodData(toY: e.value, color: Colors.blue, width: 12)],
      );
    }).toList();
  }

  // ============================================================
  // 2. Ventas por cultivo (PieChart)
  // ============================================================
  List<PieChartSectionData> getVentasPorCultivo() {
    Map<String, double> totales = {};

    for (var v in ventas) {
      final nombre = v["cultivoNombre"].toString();
      final total = (v["total"] as num).toDouble();

      totales[nombre] = (totales[nombre] ?? 0) + total;
    }

    // Paleta extendida de colores únicos para cultivos
    final cultivoColors = [
      Colors.blue.shade300,
      Colors.green.shade300,
      Colors.orange.shade300,
      Colors.purple.shade300,
      Colors.red.shade300,
      Colors.teal.shade300,
      Colors.indigo.shade300,
      Colors.pink.shade300,
      Colors.amber.shade300,
      Colors.cyan.shade300,
      Colors.lime.shade300,
      Colors.brown.shade300,
      Colors.blueGrey.shade300,
      Colors.deepOrange.shade300,
      Colors.deepPurple.shade300,
      Colors.lightBlue.shade300,
      Colors.lightGreen.shade300,
      Colors.yellow.shade300,
      Colors.lightGreen.shade400,
      Colors.orange.shade400,
      Colors.purple.shade400,
      Colors.red.shade400,
      Colors.teal.shade400,
      Colors.indigo.shade400,
      Colors.pink.shade400,
      Colors.amber.shade400,
      Colors.cyan.shade400,
      Colors.lime.shade400,
      Colors.brown.shade400,
      Colors.blueGrey.shade400,
    ];

    final usedColors = <Color>{};
    final cultivoColorMap = <String, Color>{};

    // Asignar colores únicos a cada cultivo
    for (var cultivoName in totales.keys) {
      Color selectedColor;

      // Intentar usar color basado en hash primero
      final hashIndex = cultivoName.hashCode.abs() % cultivoColors.length;
      selectedColor = cultivoColors[hashIndex];

      // Si el color ya está usado, buscar el siguiente disponible
      if (usedColors.contains(selectedColor)) {
        for (int i = 0; i < cultivoColors.length; i++) {
          final candidateColor =
              cultivoColors[(hashIndex + i) % cultivoColors.length];
          if (!usedColors.contains(candidateColor)) {
            selectedColor = candidateColor;
            break;
          }
        }
      }

      cultivoColorMap[cultivoName] = selectedColor;
      usedColors.add(selectedColor);
    }

    return totales.entries.map((e) {
      final cultivoColor = cultivoColorMap[e.key]!;

      return PieChartSectionData(
        color: cultivoColor,
        value: e.value,
        title: e.key,
        radius: 55,
        titleStyle: const TextStyle(
          color: Colors.black,
          fontWeight: FontWeight.bold,
          fontSize: 12,
        ),
      );
    }).toList();
  }

  // ============================================================
  // 3. Cantidad cosechada por cultivo
  // ============================================================
  List<BarChartGroupData> getCosechaPorCultivo() {
    int index = 0;

    return cultivos.map((cult) {
      final cant = (cult["cantidadCosechada"] as num?)?.toDouble() ?? 0.0;

      return BarChartGroupData(
        x: index++,
        barRods: [BarChartRodData(toY: cant, color: Colors.green, width: 14)],
      );
    }).toList();
  }

  // ============================================================
  // 4. Ganancias vs Gastos
  // ============================================================
  List<BarChartGroupData> getGananciasVsGastos() {
    double totalIngresos = 0;
    double totalEgresos = 0;

    for (var c in cultivos) {
      totalIngresos += (c["ingresos"] as num?)?.toDouble() ?? 0.0;
      totalEgresos += (c["egresos"] as num?)?.toDouble() ?? 0.0;
    }

    return [
      BarChartGroupData(
        x: 1,
        barRods: [
          BarChartRodData(toY: totalIngresos, color: Colors.green, width: 20),
          BarChartRodData(toY: totalEgresos, color: Colors.red, width: 20),
        ],
      ),
    ];
  }

  // ============================================================
  // 5. Gráfica de riesgos
  // ============================================================
  List<PieChartSectionData> getCultivosEnRiesgo() {
    int riesgo = cultivos.where((c) => c["isRisk"] == 1).length;
    int sanos = cultivos.length - riesgo;

    return [
      PieChartSectionData(
        color: Colors.red.shade300,
        value: riesgo.toDouble(),
        title: "Riesgo",
        radius: 55,
        titleStyle: const TextStyle(
          color: Colors.black,
          fontWeight: FontWeight.bold,
          fontSize: 12,
        ),
      ),
      PieChartSectionData(
        color: Colors.lightGreen.shade400,
        value: sanos.toDouble(),
        title: "Sanos",
        radius: 55,
        titleStyle: const TextStyle(
          color: Colors.black,
          fontWeight: FontWeight.bold,
          fontSize: 12,
        ),
      ),
    ];
  }

  // ============================================================
  // 6. Cultivos por categoría
  // ============================================================
  List<PieChartSectionData> getCultivosPorCategoria() {
    Map<String, int> categoriasCount = {};

    // Crear mapa de ID a nombre de categoría
    Map<String, String> categoriaNombres = {};
    for (var cat in categorias) {
      final id = cat["id"]?.toString() ?? "";
      final nombre = cat["nombre"]?.toString() ?? "Sin categoría";
      categoriaNombres[id] = nombre;
    }

    for (var c in cultivos) {
      final catId = c["categoriaId"]?.toString() ?? "N/A";
      categoriasCount[catId] = (categoriasCount[catId] ?? 0) + 1;
    }

    return categoriasCount.entries.map((e) {
      final nombreCategoria = categoriaNombres[e.key] ?? "Categoría ${e.key}";
      // Cada categoría tiene su propio color basado en su ID
      final categoriaColor = getColorFromId(e.key, shade: 300);

      return PieChartSectionData(
        color: categoriaColor,
        value: e.value.toDouble(),
        title: nombreCategoria,
        radius: 50,
        titleStyle: const TextStyle(
          color: Colors.black,
          fontWeight: FontWeight.bold,
          fontSize: 12,
        ),
      );
    }).toList();
  }

  // ============================================================
  // 7. Cultivos por tipo
  // ============================================================
  List<PieChartSectionData> getCultivosPorTipo() {
    Map<String, int> tiposCount = {};

    // Crear mapa de ID a nombre de tipo
    Map<String, String> tipoNombres = {};
    for (var tip in tipos) {
      final id = tip["id"]?.toString() ?? "";
      final nombre = tip["nombre"]?.toString() ?? "Sin tipo";
      tipoNombres[id] = nombre;
    }

    for (var c in cultivos) {
      final tipoId = c["tipoId"]?.toString() ?? "N/A";
      tiposCount[tipoId] = (tiposCount[tipoId] ?? 0) + 1;
    }

    return tiposCount.entries.map((e) {
      final nombreTipo = tipoNombres[e.key] ?? "Tipo ${e.key}";
      // Cada tipo tiene su propio color basado en su ID
      final tipoColor = getColorFromId(e.key, shade: 200);

      return PieChartSectionData(
        color: tipoColor,
        value: e.value.toDouble(),
        title: nombreTipo,
        radius: 50,
        titleStyle: const TextStyle(
          color: Colors.black,
          fontWeight: FontWeight.bold,
          fontSize: 12,
        ),
      );
    }).toList();
  }

  // ============================================================
  // 8. Resumen Financiero
  // ============================================================
  Widget _buildFinancialSummary() {
    return Consumer<StatsProvider>(
      builder: (context, stats, _) {
        final rentabilidad = stats.rentabilidad;
        final rentabilidadColor = rentabilidad >= 0 ? Colors.green : Colors.red;

        return Card(
          margin: const EdgeInsets.all(16),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                const Text(
                  '💰 Resumen Financiero',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: _buildFinancialCard(
                        'Ingresos',
                        'S/ ${stats.totalIngresos.toStringAsFixed(2)}',
                        Icons.trending_up,
                        Colors.green,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildFinancialCard(
                        'Egresos',
                        'S/ ${stats.totalEgresos.toStringAsFixed(2)}',
                        Icons.trending_down,
                        Colors.red,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        rentabilidadColor.withOpacity(0.1),
                        rentabilidadColor.withOpacity(0.05),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: rentabilidadColor, width: 2),
                  ),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: rentabilidadColor,
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          rentabilidad >= 0
                              ? Icons.attach_money
                              : Icons.money_off,
                          color: Colors.white,
                          size: 28,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Rentabilidad Total',
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey[700],
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'S/ ${rentabilidad.toStringAsFixed(2)}',
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: rentabilidadColor,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text(
                            '${stats.monthlyTrend} ${stats.monthlyVariation.abs()}%',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: rentabilidadColor,
                            ),
                          ),
                          Text(
                            'vs mes anterior',
                            style: TextStyle(
                              fontSize: 11,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildFinancialCard(
    String title,
    String amount,
    IconData icon,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
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
                style: TextStyle(fontSize: 13, color: Colors.grey[700]),
              ),
              Icon(icon, color: color, size: 20),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            amount,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  // ============================================================
  // WIDGET DE TARJETA GRÁFICA
  // ============================================================
  Widget card(String titulo, Widget child) {
    return Card(
      margin: const EdgeInsets.all(16),
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(
              titulo,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            SizedBox(height: 250, child: child),
          ],
        ),
      ),
    );
  }

  // ============================================================
  // UI
  // ============================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("📊 Reportes Generales")),
      body: loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              children: [
                // Resumen financiero al inicio
                _buildFinancialSummary(),

                card(
                  "Ventas por mes",
                  BarChart(
                    BarChartData(
                      barGroups: getVentasPorMes(),
                      borderData: FlBorderData(show: false),
                    ),
                  ),
                ),
                card(
                  "Ventas por cultivo",
                  PieChart(PieChartData(sections: getVentasPorCultivo())),
                ),
                card(
                  "Cantidad cosechada por cultivo",
                  BarChart(
                    BarChartData(
                      barGroups: getCosechaPorCultivo(),
                      borderData: FlBorderData(show: false),
                    ),
                  ),
                ),
                card(
                  "Ganancias vs Gastos",
                  BarChart(
                    BarChartData(
                      barGroups: getGananciasVsGastos(),
                      borderData: FlBorderData(show: false),
                    ),
                  ),
                ),
                card(
                  "Estado de Riesgo",
                  PieChart(PieChartData(sections: getCultivosEnRiesgo())),
                ),
                card(
                  "Cultivos por Categoría",
                  PieChart(PieChartData(sections: getCultivosPorCategoria())),
                ),
                card(
                  "Cultivos por Tipo",
                  PieChart(PieChartData(sections: getCultivosPorTipo())),
                ),
              ],
            ),
    );
  }
}
