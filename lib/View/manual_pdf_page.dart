import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_pdfview/flutter_pdfview.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class ManualPDFPage extends StatefulWidget {
  const ManualPDFPage({super.key});

  @override
  State<ManualPDFPage> createState() => _ManualPDFPageState();
}

class _ManualPDFPageState extends State<ManualPDFPage> {
  int _currentPage = 0;
  int _totalPages = 0;
  String? _error;
  String? _localPath;

  @override
  void initState() {
    super.initState();
    _preparePDF();
  }

  Future<void> _preparePDF() async {
    try {
      // 1. Obtener directorio temporal
      final directory = await getTemporaryDirectory();
      final path = '${directory.path}/manual_usuario.pdf';

      // 2. Verificar si ya existe
      final file = File(path);
      if (await file.exists()) {
        setState(() {
          _localPath = path;
        });
        return;
      }

      // 3. Copiar desde assets a directorio temporal
      final byteData = await rootBundle.load('assets/manual_usuario.pdf');
      final buffer = byteData.buffer.asUint8List();

      await file.writeAsBytes(buffer);

      setState(() {
        _localPath = path;
      });

      print('✅ PDF copiado exitosamente a: $path');
    } catch (e) {
      print('❌ Error al preparar PDF: $e');
      setState(() {
        _error = 'No se pudo cargar el manual: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Manual de Usuario'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        actions: [
          if (_localPath != null)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: () {
                setState(() {
                  _error = null;
                  _currentPage = 0;
                  _totalPages = 0;
                });
                _preparePDF();
              },
              tooltip: 'Recargar manual',
            ),
        ],
      ),
      body: _localPath == null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(color: Colors.blue),
                  const SizedBox(height: 16),
                  Text(
                    _error ?? 'Cargando manual de usuario...',
                    style: TextStyle(
                      color: _error != null ? Colors.red : Colors.grey[700],
                    ),
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 24),
                    ElevatedButton.icon(
                      onPressed: () {
                        setState(() {
                          _error = null;
                        });
                        _preparePDF();
                      },
                      icon: const Icon(Icons.refresh),
                      label: const Text('Reintentar'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue[700],
                        foregroundColor: Colors.white,
                      ),
                    ),
                  ],
                ],
              ),
            )
          : _error != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.error_outline, size: 64, color: Colors.red),
                  const SizedBox(height: 16),
                  Text(
                    'Error al cargar el PDF',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.red,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _error!,
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[600]),
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton.icon(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back),
                    label: const Text('Volver'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[700],
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            )
          : Column(
              children: [
                Expanded(
                  child: PDFView(
                    filePath: _localPath!,
                    enableSwipe: true,
                    swipeHorizontal: false,
                    autoSpacing: false,
                    pageFling: false,
                    onRender: (pages) {
                      setState(() {
                        _totalPages = pages!;
                      });
                      print('✅ PDF renderizado con $pages páginas');
                    },
                    onError: (error) {
                      print('❌ Error en PDFView: $error');
                      setState(() {
                        _error = 'Error al renderizar el PDF: $error';
                      });
                    },
                    onPageError: (page, error) {
                      print('❌ Error en página $page: $error');
                      setState(() {
                        _error = 'Error en la página $page: $error';
                      });
                    },
                    onViewCreated: (PDFViewController pdfViewController) {
                      print('✅ PDFViewController creado');
                    },
                    onPageChanged: (page, total) {
                      setState(() {
                        _currentPage = page!;
                      });
                    },
                  ),
                ),
                if (_totalPages > 0)
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.grey[100],
                      border: Border(top: BorderSide(color: Colors.grey[300]!)),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Página ${_currentPage + 1} de $_totalPages',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey[700],
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        Row(
                          children: [
                            Text(
                              'AMGeCCA v1.0.0',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey[600],
                              ),
                            ),
                            const SizedBox(width: 16),
                          ],
                        ),
                      ],
                    ),
                  ),
              ],
            ),
    );
  }
}
