import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';

class VideoTutorialesPage extends StatefulWidget {
  const VideoTutorialesPage({super.key});

  @override
  State<VideoTutorialesPage> createState() => _VideoTutorialesPageState();
}

class _VideoTutorialesPageState extends State<VideoTutorialesPage> {
  late VideoPlayerController _controller;
  bool _isInitialized = false;
  String? _error;

  // Lista de videos tutoriales disponibles
  final List<Map<String, String>> _tutoriales = [
    {
      'titulo': 'Introducción a AMGeCCA',
      'descripcion': 'Conoce las características principales de la aplicación',
      'duracion': '3:45',
      'video': 'assets/videos/video.mp4',
      'categoria': 'General',
    },
    {
      'titulo': 'Cómo agregar un nuevo cultivo',
      'descripcion': 'Paso a paso para registrar tus cultivos',
      'duracion': '5:20',
      'video': 'assets/videos/video.mp4',
      'categoria': 'Cultivos',
    },
    {
      'titulo': 'Gestión de egresos',
      'descripcion': 'Controla tus gastos de manera eficiente',
      'duracion': '4:15',
      'video': 'assets/videos/video.mp4',
      'categoria': 'Finanzas',
    },
    {
      'titulo': 'Generación de reportes',
      'descripcion': 'Analiza tus datos con reportes detallados',
      'duracion': '6:30',
      'video': 'assets/videos/video.mp4',
      'categoria': 'Reportes',
    },
  ];

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  Future<void> _initializeVideo() async {
    try {
      // Inicializar con el primer video de la lista
      _controller = VideoPlayerController.asset('assets/videos/video.mp4');

      await _controller.initialize();

      setState(() {
        _isInitialized = true;
      });

      print('✅ Video inicializado correctamente');
    } catch (e) {
      print('❌ Error al inicializar video: $e');
      setState(() {
        _error = 'No se pudo cargar el video: $e';
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video Tutoriales'),
        backgroundColor: Colors.red[700],
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: Icon(
              _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
            ),
            onPressed: () {
              setState(() {
                _controller.value.isPlaying
                    ? _controller.pause()
                    : _controller.play();
              });
            },
          ),
          IconButton(
            icon: const Icon(Icons.fullscreen),
            onPressed: () {
              // TODO: Implementar pantalla completa
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Pantalla completa disponible próximamente'),
                ),
              );
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Reproductor de video
          _buildVideoPlayer(),

          // Lista de tutoriales
          Expanded(child: _buildTutorialesList()),
        ],
      ),
    );
  }

  Widget _buildVideoPlayer() {
    return Container(
      height: 220,
      color: Colors.black,
      child: _isInitialized
          ? Stack(
              alignment: Alignment.center,
              children: [
                AspectRatio(
                  aspectRatio: _controller.value.aspectRatio,
                  child: VideoPlayer(_controller),
                ),
                // Controles de reproducción
                Positioned(
                  bottom: 8,
                  left: 8,
                  right: 8,
                  child: Row(
                    children: [
                      // Botón play/pause
                      IconButton(
                        onPressed: () {
                          setState(() {
                            _controller.value.isPlaying
                                ? _controller.pause()
                                : _controller.play();
                          });
                        },
                        icon: Icon(
                          _controller.value.isPlaying
                              ? Icons.pause
                              : Icons.play_arrow,
                          color: Colors.white,
                          size: 32,
                        ),
                      ),

                      // Barra de progreso
                      Expanded(
                        child: VideoProgressIndicator(
                          _controller,
                          allowScrubbing: true,
                          colors: const VideoProgressColors(
                            playedColor: Colors.red,
                            backgroundColor: Colors.white24,
                            bufferedColor: Colors.white38,
                          ),
                        ),
                      ),

                      // Tiempo
                      Text(
                        '${_formatDuration(_controller.value.position)} / ${_formatDuration(_controller.value.duration)}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            )
          : _error != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.error_outline, size: 48, color: Colors.red),
                  const SizedBox(height: 8),
                  Text(
                    'Error al cargar video',
                    style: TextStyle(color: Colors.red[700]),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _error!,
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[600], fontSize: 12),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _error = null;
                      });
                      _initializeVideo();
                    },
                    icon: const Icon(Icons.refresh),
                    label: const Text('Reintentar'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red[700],
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            )
          : const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Colors.red),
                  SizedBox(height: 16),
                  Text(
                    'Cargando video...',
                    style: TextStyle(color: Colors.white),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _buildTutorialesList() {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _tutoriales.length,
      itemBuilder: (context, index) {
        final tutorial = _tutoriales[index];
        final isPlaying =
            _isInitialized && index == 0; // Primer video está cargado

        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          elevation: 3,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          child: ListTile(
            contentPadding: const EdgeInsets.all(16),
            leading: Container(
              width: 80,
              height: 60,
              decoration: BoxDecoration(
                color: Colors.red[100],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.red[300]!),
              ),
              child: Stack(
                alignment: Alignment.center,
                children: [
                  Icon(
                    Icons.play_circle_outline,
                    color: Colors.red[700],
                    size: 32,
                  ),
                  if (isPlaying)
                    Positioned(
                      top: 4,
                      right: 4,
                      child: Container(
                        padding: const EdgeInsets.all(2),
                        decoration: const BoxDecoration(
                          color: Colors.red,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.volume_up,
                          color: Colors.white,
                          size: 12,
                        ),
                      ),
                    ),
                ],
              ),
            ),
            title: Text(
              tutorial['titulo']!,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 4),
                Text(
                  tutorial['descripcion']!,
                  style: TextStyle(color: Colors.grey[600], fontSize: 14),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.red[50],
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.red[200]!),
                      ),
                      child: Text(
                        tutorial['categoria']!,
                        style: TextStyle(
                          color: Colors.red[700],
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Icon(Icons.schedule, size: 14, color: Colors.grey[600]),
                    const SizedBox(width: 4),
                    Text(
                      tutorial['duracion']!,
                      style: TextStyle(color: Colors.grey[600], fontSize: 12),
                    ),
                  ],
                ),
              ],
            ),
            trailing: Icon(
              isPlaying ? Icons.pause_circle : Icons.play_circle_outline,
              color: Colors.red[700],
              size: 32,
            ),
            onTap: () {
              if (index == 0 && _isInitialized) {
                // Si es el primer video, solo play/pause
                setState(() {
                  _controller.value.isPlaying
                      ? _controller.pause()
                      : _controller.play();
                });
              } else {
                // Para otros videos, mostrar mensaje
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(
                      'Video "${tutorial['titulo']}" disponible próximamente',
                    ),
                    backgroundColor: Colors.red[700],
                    duration: const Duration(seconds: 2),
                  ),
                );
              }
            },
          ),
        );
      },
    );
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }
}
