// lib/View/amgecca_page.dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as Math;
import 'dart:io';
import 'package:flutter/services.dart';
import 'recomendaciones_helper.dart';
import '../services/deepseek_chat.dart';
import 'package:file_picker/file_picker.dart';
import '../Data/basedato_helper.dart';
import 'historial_conversaciones_page.dart';

// Definir cultivos disponibles
class Cultivo {
  final String id;
  final String nombre;
  final String icono;
  final String modeloPath;
  final String labelsPath;

  const Cultivo({
    required this.id,
    required this.nombre,
    required this.icono,
    required this.modeloPath,
    required this.labelsPath,
  });
}

class _Detection {
  final Rect box;
  final int classId;
  final double score;

  const _Detection({
    required this.box,
    required this.classId,
    required this.score,
  });
}

class _DetectionPainter extends CustomPainter {
  final List<_Detection> detections;
  final List<String> labels;
  final Size imageSize;

  _DetectionPainter({
    required this.detections,
    required this.labels,
    required this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width <= 0 || imageSize.height <= 0) return;
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    for (final detection in detections) {
      final rect = Rect.fromLTRB(
        detection.box.left * scaleX,
        detection.box.top * scaleY,
        detection.box.right * scaleX,
        detection.box.bottom * scaleY,
      );

      final color = _colorForClass(detection.classId);
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      canvas.drawRect(rect, paint);

      final label = detection.classId < labels.length
          ? labels[detection.classId]
          : 'Clase ${detection.classId}';
      final text = '$label ${(detection.score * 100).toStringAsFixed(1)}%';
      final textSpan = TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.w600,
          backgroundColor: Colors.white.withOpacity(0.7),
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      final offset = Offset(rect.left, rect.top - textPainter.height - 4);
      textPainter.paint(canvas, offset);
    }
  }

  Color _colorForClass(int classId) {
    const colors = [
      Colors.red,
      Colors.green,
      Colors.orange,
      Colors.blue,
      Colors.purple,
    ];
    return colors[classId % colors.length];
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.labels != labels ||
        oldDelegate.imageSize != imageSize;
  }
}

class _LetterboxResult {
  final img.Image image;
  final double gain;
  final double padX;
  final double padY;

  const _LetterboxResult({
    required this.image,
    required this.gain,
    required this.padX,
    required this.padY,
  });
}

class ReportesPage extends StatefulWidget {
  const ReportesPage({super.key});

  @override
  State<ReportesPage> createState() => _ReportesPageState();
}

class _ReportesPageState extends State<ReportesPage> {
  static const Cultivo _cultivoCacao = Cultivo(
    id: 'cacao',
    nombre: 'Cacao',
    icono: '🍫',
    modeloPath: 'assets/models/best.tflite',
    labelsPath: 'assets/models/labels.txt',
  );

  Cultivo? _cultivoSeleccionado;
  File? _image;
  bool _loading = false;
  bool _modeloListo = false;
  String? _resultado;
  double? _confianza;
  Interpreter? _interpreter;
  List<String> _labels = [];
  List<_Detection> _detecciones = [];
  Size? _imageSize;
  double _letterboxGain = 1.0;
  double _letterboxPadX = 0.0;
  double _letterboxPadY = 0.0;
  int _inputWidth = 224;
  int _inputHeight = 224;
  int _inputChannels = 3;
  bool _inputIsNchw = false;

  @override
  void initState() {
    super.initState();
    cargarModelo(_cultivoCacao);
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  Future<void> cargarModelo(Cultivo cultivo) async {
    setState(() {
      _cultivoSeleccionado = cultivo;
      _modeloListo = false;
      _resultado = null;
      _confianza = null;
      _detecciones = [];
      _image = null;
      _imageSize = null;
    });

    _interpreter?.close();
    _interpreter = null;

    try {
      final modelData = await rootBundle.load('assets/models/best.tflite');
      _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());
      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape;
      if (inputShape.length == 4) {
        if (inputShape[1] == 3) {
          _inputIsNchw = true;
          _inputChannels = inputShape[1];
          _inputHeight = inputShape[2];
          _inputWidth = inputShape[3];
        } else {
          _inputIsNchw = false;
          _inputHeight = inputShape[1];
          _inputWidth = inputShape[2];
          _inputChannels = inputShape[3];
        }
      }

      final labelsData = await rootBundle.loadString(cultivo.labelsPath);
      _labels = labelsData
          .split('\n')
          .map((l) => l.replaceAll(RegExp(r'^\d+\s*'), '').trim())
          .where((l) => l.isNotEmpty)
          .toList();

      setState(() {
        _modeloListo = true;
      });

      debugPrint(
        '✅ Modelo ${cultivo.nombre} cargado: listo para detectar moniliasis',
      );
      debugPrint('🔍 Input shape: $inputShape');
    } catch (e) {
      debugPrint('❌ Error cargando ${cultivo.nombre}: $e');
      setState(() {
        _resultado = "Error cargando modelo de ${cultivo.nombre}";
        _modeloListo = false;
      });
    }
  }

  Future<void> seleccionarImagen(ImageSource source) async {
    final picker = ImagePicker();
    final XFile? foto = await picker.pickImage(source: source);

    if (foto != null) {
      setState(() {
        _image = File(foto.path);
        _loading = true;
        _resultado = null;
        _confianza = null;
        _detecciones = [];
      });
      await analizarImagen(File(foto.path));
    }
  }

  Future<void> analizarImagen(File imageFile) async {
    if (_interpreter == null) {
      setState(() {
        _loading = false;
        _resultado = "Modelo no disponible";
      });
      return;
    }

    try {
      final bytes = await imageFile.readAsBytes();
      final image = img.decodeImage(bytes);

      if (image == null) {
        setState(() {
          _loading = false;
          _resultado = "No se pudo leer la imagen";
        });
        return;
      }

      final letterbox = _letterboxResize(image, _inputWidth, _inputHeight);
      final resized = letterbox.image;
      _letterboxGain = letterbox.gain;
      _letterboxPadX = letterbox.padX;
      _letterboxPadY = letterbox.padY;

      dynamic input;
      if (_inputIsNchw) {
        input = List.generate(
          1,
          (_) => List.generate(
            _inputChannels,
            (c) => List.generate(_inputHeight, (y) {
              return List.generate(_inputWidth, (x) {
                final pixel = resized.getPixel(x, y);
                if (c == 0) return pixel.r / 255.0;
                if (c == 1) return pixel.g / 255.0;
                return pixel.b / 255.0;
              });
            }),
          ),
        );
      } else {
        input = List.generate(
          1,
          (_) => List.generate(
            _inputHeight,
            (y) => List.generate(_inputWidth, (x) {
              final pixel = resized.getPixel(x, y);
              return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
            }),
          ),
        );
      }

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      final outputSize = outputShape.fold<int>(1, (a, b) => a * b);
      var output = List.filled(outputSize, 0.0).reshape(outputShape);

      _interpreter!.run(input, output);

      int maxIdx = -1;
      double maxConf = 0.0;
      final detecciones = <_Detection>[];

      if (outputShape.length == 2 && outputShape[1] == _labels.length) {
        final results = (output[0] as List).cast<double>();
        final normalized = _normalizeScores(results);
        for (int i = 0; i < normalized.length; i++) {
          if (normalized[i] > maxConf) {
            maxConf = normalized[i];
            maxIdx = i;
          }
        }
      } else if (outputShape.length == 3 &&
          outputShape[2] >= 5 + _labels.length) {
        final detections = _normalizeDetectionOutput(output, outputShape);
        for (final det in detections) {
          if (det is! List || det.length < 5 + _labels.length) continue;
          final objectness = _probability((det[4] as num).toDouble());
          int bestClass = -1;
          double bestScore = 0.0;
          for (int c = 0; c < _labels.length; c++) {
            final classScore = _probability((det[5 + c] as num).toDouble());
            final score = objectness * classScore;
            if (score > bestScore) {
              bestScore = score;
              bestClass = c;
            }
          }
          if (bestClass == -1 || bestScore < 0.35) continue;
          final box = _decodeYoloBox(
            det,
            image.width.toDouble(),
            image.height.toDouble(),
          );
          if (box == null) continue;
          detecciones.add(
            _Detection(box: box, classId: bestClass, score: bestScore),
          );
        }
        final nmsDetections = _applyNms(detecciones, 0.45);
        if (nmsDetections.isNotEmpty) {
          nmsDetections.sort((a, b) => b.score.compareTo(a.score));
          maxConf = nmsDetections.first.score;
          maxIdx = nmsDetections.first.classId;
          detecciones
            ..clear()
            ..addAll(nmsDetections);
        }
      }

      setState(() {
        _loading = false;
        _confianza = maxIdx == -1 ? null : maxConf;
        _detecciones = detecciones;
        _imageSize = Size(image.width.toDouble(), image.height.toDouble());
        _resultado = maxIdx >= 0 && maxIdx < _labels.length
            ? _labels[maxIdx]
            : detecciones.isNotEmpty
                ? "Clase desconocida"
                : "Sin detecciones";
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _resultado = "Error: $e";
      });
    }
  }

  Color _getColorConfianza(double conf) {
    if (conf >= 0.8) return Colors.green;
    if (conf >= 0.5) return Colors.orange;
    return Colors.red;
  }

  _LetterboxResult _letterboxResize(img.Image image, int width, int height) {
    final gain =
        Math.min(width / image.width.toDouble(), height / image.height.toDouble());
    final newWidth = (image.width * gain).round();
    final newHeight = (image.height * gain).round();
    final resized = img.copyResize(image, width: newWidth, height: newHeight);
    final canvas = img.Image(width: width, height: height);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
    final padX = ((width - newWidth) / 2).round();
    final padY = ((height - newHeight) / 2).round();
    img.compositeImage(canvas, resized, dstX: padX, dstY: padY);
    return _LetterboxResult(
      image: canvas,
      gain: gain,
      padX: padX.toDouble(),
      padY: padY.toDouble(),
    );
  }

  double _probability(double value) {
    if (value >= 0.0 && value <= 1.0) {
      return value;
    }
    return _sigmoid(value);
  }

  double _sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
  }

  List<List<double>> _normalizeDetectionOutput(
    List output,
    List<int> outputShape,
  ) {
    final raw = output[0] as List;
    if (outputShape.length != 3) {
      return raw
          .map(
            (row) => (row as List)
                .map((value) => (value as num).toDouble())
                .toList(),
          )
          .toList();
    }
    final dim1 = outputShape[1];
    final dim2 = outputShape[2];
    if (dim1 < dim2) {
      return List.generate(
        dim2,
        (i) => List.generate(
          dim1,
          (j) => ((raw[j] as List)[i] as num).toDouble(),
        ),
      );
    }
    return raw
        .map(
          (row) => (row as List)
              .map((value) => (value as num).toDouble())
              .toList(),
        )
        .toList();
  }

  List<double> _normalizeScores(List<double> scores) {
    final needsNormalization =
        scores.any((value) => value < 0 || value > 1);
    if (!needsNormalization) {
      return scores;
    }
    final maxScore = scores.reduce((a, b) => a > b ? a : b);
    final expScores = scores.map((s) => Math.exp(s - maxScore)).toList();
    final sumExp = expScores.fold<double>(0.0, (a, b) => a + b);
    return expScores.map((s) => s / sumExp).toList();
  }

  Rect? _decodeYoloBox(List det, double imageWidth, double imageHeight) {
    final rawX = (det[0] as num).toDouble();
    final rawY = (det[1] as num).toDouble();
    final rawW = (det[2] as num).toDouble();
    final rawH = (det[3] as num).toDouble();

    if (rawW <= 0 || rawH <= 0) return null;

    double x = rawX;
    double y = rawY;
    double w = rawW;
    double h = rawH;

    if (x >= 0 && x <= 1 && y >= 0 && y <= 1 && w <= 1 && h <= 1) {
      x *= _inputWidth;
      y *= _inputHeight;
      w *= _inputWidth;
      h *= _inputHeight;
    }

    final left = x - w / 2;
    final top = y - h / 2;
    final right = x + w / 2;
    final bottom = y + h / 2;

    final mappedLeft = (left - _letterboxPadX) / _letterboxGain;
    final mappedTop = (top - _letterboxPadY) / _letterboxGain;
    final mappedRight = (right - _letterboxPadX) / _letterboxGain;
    final mappedBottom = (bottom - _letterboxPadY) / _letterboxGain;

    final rect = Rect.fromLTRB(
      mappedLeft.clamp(0.0, imageWidth),
      mappedTop.clamp(0.0, imageHeight),
      mappedRight.clamp(0.0, imageWidth),
      mappedBottom.clamp(0.0, imageHeight),
    );

    if (rect.width <= 1 || rect.height <= 1) return null;
    return rect;
  }

  List<_Detection> _applyNms(List<_Detection> detections, double iouThreshold) {
    if (detections.isEmpty) return [];
    final sorted = List<_Detection>.from(detections)
      ..sort((a, b) => b.score.compareTo(a.score));
    final selected = <_Detection>[];
    while (sorted.isNotEmpty) {
      final current = sorted.removeAt(0);
      selected.add(current);
      sorted.removeWhere(
        (candidate) => _iou(current.box, candidate.box) > iouThreshold,
      );
    }
    return selected;
  }

  double _iou(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.isEmpty) return 0.0;
    final intersectionArea = intersection.width * intersection.height;
    final unionArea =
        (a.width * a.height) + (b.width * b.height) - intersectionArea;
    if (unionArea <= 0) return 0.0;
    return intersectionArea / unionArea;
  }

  Widget _buildRecomendaciones() {
    if (_resultado == null || _cultivoSeleccionado == null) {
      return const SizedBox.shrink();
    }

    String claveEnfermedad = '${_cultivoSeleccionado!.nombre}_$_resultado';
    final recomendacion = RecomendacionesHelper.obtenerRecomendacion(
      claveEnfermedad,
    );

    if (recomendacion == null) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Text(
          'No hay recomendaciones disponibles para esta detección.',
          style: TextStyle(color: Colors.grey),
          textAlign: TextAlign.center,
        ),
      );
    }

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: recomendacion.color.withOpacity(0.1),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(12),
                topRight: Radius.circular(12),
              ),
            ),
            child: Row(
              children: [
                Icon(recomendacion.icono, color: recomendacion.color, size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    recomendacion.enfermedad,
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: recomendacion.color,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  recomendacion.descripcion,
                  style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                ),
                const SizedBox(height: 16),
                _buildSeccion(
                  'Síntomas',
                  Icons.visibility,
                  recomendacion.sintomas,
                  Colors.orange,
                ),
                const SizedBox(height: 12),
                _buildSeccion(
                  'Tratamientos Recomendados',
                  Icons.medical_services,
                  recomendacion.tratamientos,
                  Colors.red,
                ),
                const SizedBox(height: 12),
                _buildSeccion(
                  'Prevención',
                  Icons.shield,
                  recomendacion.prevencion,
                  Colors.green,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSeccion(
    String titulo,
    IconData icono,
    List<String> items,
    Color color,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icono, size: 20, color: color),
            const SizedBox(width: 8),
            Text(
              titulo,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ...items.map(
          (item) => Padding(
            padding: const EdgeInsets.only(left: 28, bottom: 6),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('• ', style: TextStyle(color: color, fontSize: 16)),
                Expanded(
                  child: Text(
                    item,
                    style: const TextStyle(fontSize: 14, height: 1.4),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  void _mostrarChatBot() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      isDismissible: true,
      enableDrag: true,
      builder: (context) => const ChatBotModal(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AMGeCCA IA'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        children: [
          SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (_cultivoSeleccionado != null)
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: _modeloListo
                            ? Colors.green[50]
                            : Colors.orange[50],
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            _modeloListo
                                ? Icons.check_circle
                                : Icons.hourglass_empty,
                            color: _modeloListo ? Colors.green : Colors.orange,
                            size: 20,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            _modeloListo
                                ? '${_cultivoSeleccionado!.nombre}: detección de moniliasis activado'
                                : 'Cargando modelo...',
                            style: TextStyle(
                              color: _modeloListo
                                  ? Colors.green[800]
                                  : Colors.orange[800],
                            ),
                          ),
                        ],
                      ),
                    ),
                  const SizedBox(height: 20),
                  Container(
                    height: 220,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Colors.grey[200],
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey[300]!),
                    ),
                    child: _image != null
                        ? ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                final imageSize =
                                    _imageSize ?? const Size(1, 1);
                                final fitted = applyBoxFit(
                                  BoxFit.contain,
                                  imageSize,
                                  constraints.biggest,
                                );
                                final renderSize = fitted.destination;
                                final dx =
                                    (constraints.maxWidth - renderSize.width) /
                                        2;
                                final dy =
                                    (constraints.maxHeight - renderSize.height) /
                                        2;

                                return Stack(
                                  children: [
                                    Positioned(
                                      left: dx,
                                      top: dy,
                                      width: renderSize.width,
                                      height: renderSize.height,
                                      child: Image.file(
                                        _image!,
                                        fit: BoxFit.contain,
                                      ),
                                    ),
                                    Positioned(
                                      left: dx,
                                      top: dy,
                                      width: renderSize.width,
                                      height: renderSize.height,
                                      child: CustomPaint(
                                        painter: _DetectionPainter(
                                          detections: _detecciones,
                                          labels: _labels,
                                          imageSize: imageSize,
                                        ),
                                      ),
                                    ),
                                  ],
                                );
                              },
                            ),
                          )
                        : Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.image,
                                size: 50,
                                color: Colors.grey[400],
                              ),
                              const SizedBox(height: 8),
                              Text(
                                _modeloListo
                                    ? 'Selecciona una imagen'
                                    : 'Cargando modelo...',
                                style: TextStyle(color: Colors.grey[600]),
                              ),
                            ],
                          ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _modeloListo
                              ? () => seleccionarImagen(ImageSource.camera)
                              : null,
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Cámara'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _modeloListo
                              ? () => seleccionarImagen(ImageSource.gallery)
                              : null,
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Galería'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green[700],
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  if (_loading)
                    const Center(
                      child: Column(
                        children: [
                          CircularProgressIndicator(color: Colors.green),
                          SizedBox(height: 12),
                          Text('Analizando imagen...'),
                        ],
                      ),
                    ),
                  if (_resultado != null && !_loading)
                    Column(
                      children: [
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(12),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.1),
                                blurRadius: 10,
                                offset: const Offset(0, 4),
                              ),
                            ],
                          ),
                          child: Column(
                            children: [
                              Text(
                                _cultivoSeleccionado?.icono ?? '🌱',
                                style: const TextStyle(fontSize: 40),
                              ),
                              const SizedBox(height: 8),
                              const Text(
                                'Resultado del análisis',
                                style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.grey,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                _resultado!,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              if (_confianza != null) ...[
                                const SizedBox(height: 12),
                                LinearProgressIndicator(
                                  value: _confianza,
                                  backgroundColor: Colors.grey[200],
                                  color: _getColorConfianza(_confianza!),
                                  minHeight: 8,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Confianza: ${(_confianza! * 100).toStringAsFixed(1)}%',
                                  style: TextStyle(
                                    color: _getColorConfianza(_confianza!),
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        _buildRecomendaciones(),
                      ],
                    ),
                  const SizedBox(height: 80),
                ],
              ),
            ),
          ),

          // Botón flotante del ChatBot
          Positioned(
            right: 16,
            bottom: 16,
            child: FloatingActionButton.extended(
              heroTag: 'chatbot_fab',
              onPressed: _mostrarChatBot,
              backgroundColor: Colors.deepPurple,
              icon: const Icon(Icons.reviews_rounded, color: Colors.white),
              label: const Text(
                "ChatBot",
                style: TextStyle(color: Colors.white),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Modal del ChatBot con soporte de archivos
// 🔥 DESDE LÍNEA 765 - Modal del ChatBot con historial
class ChatBotModal extends StatefulWidget {
  final int? conversacionId;

  const ChatBotModal({Key? key, this.conversacionId}) : super(key: key);

  @override
  State<ChatBotModal> createState() => _ChatBotModalState();
}

class _ChatBotModalState extends State<ChatBotModal> {
  final DeepSeekChat bot = DeepSeekChat();
  final BasedatoHelper db = BasedatoHelper.instance;
  final TextEditingController controller = TextEditingController();
  final ScrollController scrollController = ScrollController();

  List<Map<String, dynamic>> messages = [];
  bool isLoading = false;
  bool showAttachmentMenu = false;
  int? conversacionActualId;
  bool cargandoHistorial = false;

  @override
  void initState() {
    super.initState();
    _inicializarConversacion();
  }

  Future<void> _inicializarConversacion() async {
    if (widget.conversacionId != null) {
      setState(() => cargandoHistorial = true);
      conversacionActualId = widget.conversacionId;
      await _cargarMensajes();
      setState(() => cargandoHistorial = false);
    } else {
      conversacionActualId = await db.crearConversacion('Nueva conversación');
    }
  }

  Future<void> _cargarMensajes() async {
    if (conversacionActualId == null) return;

    final mensajesDb = await db.getMensajes(conversacionActualId!);

    setState(() {
      messages = mensajesDb.map((msg) {
        final tipo = msg['tipo'] as String;

        if (tipo == 'user_text') {
          return {'user': msg['contenido'], 'type': 'text'};
        } else if (tipo == 'bot_text') {
          return {'bot': msg['contenido'], 'type': 'text'};
        } else if (tipo == 'user_image') {
          return {
            'user': msg['archivoNombre'] ?? '📷 Imagen',
            'type': 'image',
            'file': File(msg['archivoPath'] as String),
          };
        } else if (tipo == 'user_document') {
          return {
            'user': msg['archivoNombre'] ?? '📄 Documento',
            'type': 'document',
            'file': File(msg['archivoPath'] as String),
            'fileName': msg['archivoNombre'],
          };
        }

        return {'bot': msg['contenido'], 'type': 'text'};
      }).toList();
    });

    _scrollToBottom();
  }

  Future<void> _guardarMensaje(
    String tipo,
    String contenido, {
    String? archivoPath,
    String? archivoNombre,
    String? archivoTipo,
  }) async {
    if (conversacionActualId == null) return;

    await db.insertarMensaje({
      'conversacionId': conversacionActualId,
      'tipo': tipo,
      'contenido': contenido,
      'fecha': DateTime.now().toIso8601String(),
      'archivoPath': archivoPath,
      'archivoNombre': archivoNombre,
      'archivoTipo': archivoTipo,
    });

    if (messages.length == 1 && tipo == 'user_text') {
      final titulo = contenido.length > 50
          ? '${contenido.substring(0, 50)}...'
          : contenido;
      await db.actualizarTituloConversacion(conversacionActualId!, titulo);
    }
  }

  @override
  void dispose() {
    controller.dispose();
    scrollController.dispose();
    super.dispose();
  }

  void send() async {
    String text = controller.text.trim();
    if (text.isEmpty) return;

    setState(() {
      messages.add({'user': text, 'type': 'text'});
      isLoading = true;
    });

    controller.clear();
    _scrollToBottom();

    await _guardarMensaje('user_text', text);

    String reply = await bot.sendMessage(text);

    setState(() {
      messages.add({'bot': reply, 'type': 'text'});
      isLoading = false;
    });

    await _guardarMensaje('bot_text', reply);
    _scrollToBottom();
  }

  void pickImage() async {
    setState(() => showAttachmentMenu = false);

    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 512,
      maxHeight: 512,
      imageQuality: 70,
    );

    if (image != null) {
      final imageFile = File(image.path);

      setState(() {
        messages.add({
          'user': '📷 Imagen adjunta',
          'type': 'image',
          'file': imageFile,
        });
        isLoading = true;
      });
      _scrollToBottom();

      await _guardarMensaje(
        'user_image',
        '📷 Imagen adjunta',
        archivoPath: image.path,
        archivoNombre: '📷 Imagen',
        archivoTipo: 'image',
      );

      String reply = await bot.sendMessageWithImage(
        'Analiza esta imagen relacionada con cultivos agrícolas',
        imageFile,
      );

      setState(() {
        messages.add({'bot': reply, 'type': 'text'});
        isLoading = false;
      });

      await _guardarMensaje('bot_text', reply);
      _scrollToBottom();
    }
  }

  void pickDocument() async {
    setState(() => showAttachmentMenu = false);

    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'txt', 'csv'],
    );

    if (result != null && result.files.single.path != null) {
      final file = File(result.files.single.path!);
      final fileName = result.files.single.name;

      setState(() {
        messages.add({
          'user': '📄 $fileName',
          'type': 'document',
          'file': file,
          'fileName': fileName,
        });
        isLoading = true;
      });
      _scrollToBottom();

      await _guardarMensaje(
        'user_document',
        '📄 Documento adjunto',
        archivoPath: file.path,
        archivoNombre: fileName,
        archivoTipo: 'document',
      );

      String reply = await bot.sendMessageWithDocument(
        'Analiza este documento',
        file,
        fileName,
      );

      setState(() {
        messages.add({'bot': reply, 'type': 'text'});
        isLoading = false;
      });

      await _guardarMensaje('bot_text', reply);
      _scrollToBottom();
    }
  }

  void takePhoto() async {
    setState(() => showAttachmentMenu = false);

    final picker = ImagePicker();
    final XFile? photo = await picker.pickImage(
      source: ImageSource.camera,
      maxWidth: 512,
      maxHeight: 512,
      imageQuality: 70,
    );

    if (photo != null) {
      final photoFile = File(photo.path);

      setState(() {
        messages.add({
          'user': '📸 Foto tomada',
          'type': 'image',
          'file': photoFile,
        });
        isLoading = true;
      });
      _scrollToBottom();

      await _guardarMensaje(
        'user_image',
        '📸 Foto tomada',
        archivoPath: photo.path,
        archivoNombre: '📸 Foto',
        archivoTipo: 'image',
      );

      String reply = await bot.sendMessageWithImage(
        'Analiza esta foto relacionada con cultivos agrícolas',
        photoFile,
      );

      setState(() {
        messages.add({'bot': reply, 'type': 'text'});
        isLoading = false;
      });

      await _guardarMensaje('bot_text', reply);
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 150), () {
      if (scrollController.hasClients) {
        scrollController.animateTo(
          scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _mostrarHistorial() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => HistorialConversacionesPage(
          onSeleccionarConversacion: (conversacionId) {
            Navigator.of(context).pop();
            Navigator.of(context).pop();
            showModalBottomSheet(
              context: context,
              isScrollControlled: true,
              backgroundColor: Colors.transparent,
              isDismissible: true,
              enableDrag: true,
              builder: (context) =>
                  ChatBotModal(conversacionId: conversacionId),
            );
          },
        ),
      ),
    );
  }

  Widget _buildMessage(Map<String, dynamic> msg, bool isBot) {
    if (msg['type'] == 'image' && !isBot) {
      return Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(8),
        constraints: const BoxConstraints(maxWidth: 200),
        decoration: BoxDecoration(
          color: Colors.deepPurple[100],
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (msg['file'] != null && File(msg['file'].path).existsSync())
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(
                  msg['file'],
                  height: 150,
                  width: 150,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      height: 150,
                      width: 150,
                      color: Colors.grey[300],
                      child: const Icon(Icons.broken_image, size: 50),
                    );
                  },
                ),
              ),
            const SizedBox(height: 4),
            Text(msg['user'], style: const TextStyle(fontSize: 12)),
          ],
        ),
      );
    }

    if (msg['type'] == 'document' && !isBot) {
      return Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(12),
        constraints: const BoxConstraints(maxWidth: 250),
        decoration: BoxDecoration(
          color: Colors.deepPurple[100],
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.insert_drive_file,
              size: 32,
              color: Colors.deepPurple,
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                msg['user'],
                style: const TextStyle(fontSize: 14),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      );
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(12),
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.75,
      ),
      decoration: BoxDecoration(
        color: isBot ? Colors.grey[200] : Colors.deepPurple[100],
        borderRadius: BorderRadius.circular(16),
      ),
      child: Text(
        isBot ? msg['bot'] : msg['user'],
        style: const TextStyle(fontSize: 15),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedPadding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      duration: const Duration(milliseconds: 100),
      child: Container(
        height: MediaQuery.of(context).size.height * 0.75,
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(24),
            topRight: Radius.circular(24),
          ),
        ),
        child: Column(
          children: [
            // Header con botón hamburguesa
            Container(
              padding: const EdgeInsets.all(16),
              decoration: const BoxDecoration(
                color: Colors.deepPurple,
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(24),
                  topRight: Radius.circular(24),
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.menu, color: Colors.white, size: 28),
                    onPressed: _mostrarHistorial,
                    tooltip: 'Historial de conversaciones',
                  ),
                  const SizedBox(width: 4),
                  const Icon(Icons.smart_toy, color: Colors.white, size: 28),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Asistente IA',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'AMGeCCA',
                          style: TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),

            // Messages
            Expanded(
              child: cargandoHistorial
                  ? const Center(
                      child: CircularProgressIndicator(
                        color: Colors.deepPurple,
                      ),
                    )
                  : messages.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.chat_bubble_outline,
                            size: 64,
                            color: Colors.grey[300],
                          ),
                          const SizedBox(height: 16),
                          Text(
                            '¡Hola! Soy tu asistente agrícola',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey[600],
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Pregúntame sobre cultivos, plagas o enfermedades',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey[500],
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 10,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.deepPurple[50],
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(
                                  Icons.history,
                                  size: 18,
                                  color: Colors.deepPurple,
                                ),
                                const SizedBox(width: 8),
                                Text(
                                  'Tus chats se guardan automáticamente',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.deepPurple[700],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      controller: scrollController,
                      padding: const EdgeInsets.all(16),
                      itemCount: messages.length + (isLoading ? 1 : 0),
                      itemBuilder: (_, index) {
                        if (index == messages.length && isLoading) {
                          return Align(
                            alignment: Alignment.centerLeft,
                            child: Container(
                              margin: const EdgeInsets.symmetric(vertical: 6),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.grey[200],
                                borderRadius: BorderRadius.circular(16),
                              ),
                              child: const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  ),
                                  SizedBox(width: 8),
                                  Text('Analizando...'),
                                ],
                              ),
                            ),
                          );
                        }

                        final msg = messages[index];
                        final isBot = msg.containsKey('bot');

                        return Align(
                          alignment: isBot
                              ? Alignment.centerLeft
                              : Alignment.centerRight,
                          child: _buildMessage(msg, isBot),
                        );
                      },
                    ),
            ),

            // Menú de adjuntos
            if (showAttachmentMenu)
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  border: Border(top: BorderSide(color: Colors.grey[300]!)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildAttachmentButton(
                      icon: Icons.camera_alt,
                      label: 'Cámara',
                      color: Colors.blue,
                      onTap: takePhoto,
                    ),
                    _buildAttachmentButton(
                      icon: Icons.photo_library,
                      label: 'Galería',
                      color: Colors.green,
                      onTap: pickImage,
                    ),
                    _buildAttachmentButton(
                      icon: Icons.insert_drive_file,
                      label: 'Documento',
                      color: Colors.orange,
                      onTap: pickDocument,
                    ),
                  ],
                ),
              ),

            // Input
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.white,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 10,
                    offset: const Offset(0, -2),
                  ),
                ],
              ),
              child: SafeArea(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    IconButton(
                      icon: Icon(
                        showAttachmentMenu ? Icons.close : Icons.add_circle,
                        color: Colors.deepPurple,
                        size: 28,
                      ),
                      onPressed: () {
                        setState(() {
                          showAttachmentMenu = !showAttachmentMenu;
                        });
                      },
                    ),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Container(
                        constraints: const BoxConstraints(maxHeight: 120),
                        child: TextField(
                          controller: controller,
                          maxLines: null,
                          keyboardType: TextInputType.multiline,
                          textInputAction: TextInputAction.send,
                          decoration: InputDecoration(
                            hintText: 'Escribe tu consulta...',
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(24),
                              borderSide: BorderSide.none,
                            ),
                            filled: true,
                            fillColor: Colors.grey[100],
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 12,
                            ),
                          ),
                          onSubmitted: (_) => send(),
                        ),
                      ),
                    ),
                    const SizedBox(width: 4),
                    Container(
                      decoration: const BoxDecoration(
                        color: Colors.deepPurple,
                        shape: BoxShape.circle,
                      ),
                      child: IconButton(
                        icon: const Icon(Icons.send, color: Colors.white),
                        onPressed: send,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAttachmentButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: color, size: 28),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                color: color,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
