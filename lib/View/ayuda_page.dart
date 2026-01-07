import 'package:flutter/material.dart';
import 'package:amgeca/View/manual_pdf_page.dart';
import 'package:amgeca/View/videotutoriales_page.dart';

class AyudaPage extends StatelessWidget {
  const AyudaPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Centro de Ayuda'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '¿En qué podemos ayudarte?',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Encuentra respuestas y recursos para usar AMGeCCA de manera efectiva',
              style: TextStyle(fontSize: 16, color: Colors.grey[600]),
            ),
            const SizedBox(height: 32),

            // Manual de Usuario
            _buildOpcionAyuda(
              icon: Icons.picture_as_pdf,
              title: 'Manual de Usuario',
              subtitle: 'Guía completa de AMGeCCA',
              color: Colors.blue,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const ManualPDFPage()),
                );
              },
            ),

            const SizedBox(height: 16),

            // Tutoriales
            _buildOpcionAyuda(
              icon: Icons.video_library,
              title: 'Tutoriales en Video',
              subtitle: 'Aprende usando AMGeCCA paso a paso',
              color: Colors.red,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => const VideoTutorialesPage(),
                  ),
                );
              },
            ),

            const SizedBox(height: 16),

            // Soporte Técnico
            _buildOpcionAyuda(
              icon: Icons.support_agent,
              title: 'Soporte Técnico',
              subtitle: 'Contacta con nuestro equipo',
              color: Colors.green,
              onTap: () {
                _mostrarSoporte(context);
              },
            ),

            const SizedBox(height: 16),

            // Preguntas Frecuentes
            _buildOpcionAyuda(
              icon: Icons.question_answer,
              title: 'Preguntas Frecuentes',
              subtitle: 'Respuestas a dudas comunes',
              color: Colors.orange,
              onTap: () {
                _mostrarFAQ(context);
              },
            ),

            const SizedBox(height: 16),

            // Sugerencias
            _buildOpcionAyuda(
              icon: Icons.lightbulb_outline,
              title: 'Enviar Sugerencias',
              subtitle: 'Ayúdanos a mejorar AMGeCCA',
              color: Colors.purple,
              onTap: () {
                _enviarSugerencia(context);
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildOpcionAyuda({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 32),
            ),
            const SizedBox(width: 20),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                      color: color,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios,
              color: color.withOpacity(0.6),
              size: 24,
            ),
          ],
        ),
      ),
    );
  }

  void _mostrarSoporte(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(Icons.support_agent, color: Colors.green[700]),
            const SizedBox(width: 12),
            const Text('Soporte Técnico'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('¿Necesitas ayuda? Contáctanos:'),
            const SizedBox(height: 16),
            _buildContactoItem(
              icon: Icons.email,
              label: 'Email',
              value: 'amgecca@gmail.com',
              onTap: () => _copiarAlPortapapeles(context, 'amgecca@gmail.com'),
            ),
            const SizedBox(height: 12),
            _buildContactoItem(
              icon: Icons.phone,
              label: 'Teléfono',
              value: '+51 957 492 678',
              onTap: () => _copiarAlPortapapeles(context, '+51 957 492 678'),
            ),
            const SizedBox(height: 12),
            _buildContactoItem(
              icon: Icons.schedule,
              label: 'Horario',
              value: 'Lun-Vie 9:00-17:00',
              onTap: null,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cerrar'),
          ),
        ],
      ),
    );
  }

  Widget _buildContactoItem({
    required IconData icon,
    required String label,
    required String value,
    required VoidCallback? onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.grey[50],
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.grey[200]!),
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.green[700], size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  Text(
                    value,
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            if (onTap != null)
              Icon(Icons.copy, color: Colors.green[700], size: 18),
          ],
        ),
      ),
    );
  }

  void _mostrarFAQ(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(Icons.question_answer, color: Colors.orange[700]),
            const SizedBox(width: 12),
            const Text('Preguntas Frecuentes'),
          ],
        ),
        content: SizedBox(
          width: double.maxFinite,
          height: 400,
          child: ListView(
            children: [
              _buildFAQItem(
                pregunta: '¿Cómo agrego un nuevo tipo de cultivo?',
                respuesta:
                    'Ve a la sección Cultivos en la parte superior derecha presiona en los tres puntos ahi podrás agregar un nuevo tipo de cultivo.',
              ),
              _buildFAQItem(
                pregunta: '¿Cómo agrego una nueva categoría?',
                respuesta:
                    'Ve a la sección Cultivos en la parte superior derecha presiona en los tres puntos ahi podrás agregar una nueva categoría.',
              ),
              _buildFAQItem(
                pregunta: '¿Cómo agrego un nuevo cultivo?',
                respuesta:
                    'Ve a la sección Cultivos y presiona el botón + para agregar un nuevo cultivo.',
              ),
              _buildFAQItem(
                pregunta: '¿Cómo registro egresos?',
                respuesta:
                    'Desde la lista de cultivos, selecciona un cultivo activo y presiona el botón de egresos (💰).',
              ),
              _buildFAQItem(
                pregunta: '¿Cómo genero reportes?',
                respuesta:
                    'Accede al menú hamburguesa y selecciona Reportes para ver estadísticas detalladas.',
              ),
              _buildFAQItem(
                pregunta: '¿Cómo respaldo mis datos?',
                respuesta:
                    'Ve a Ajustes > Respaldo de datos para crear una copia de seguridad.',
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cerrar'),
          ),
        ],
      ),
    );
  }

  Widget _buildFAQItem({required String pregunta, required String respuesta}) {
    return ExpansionTile(
      title: Text(
        pregunta,
        style: const TextStyle(fontWeight: FontWeight.w600),
      ),
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Text(respuesta, style: TextStyle(color: Colors.grey[700])),
        ),
      ],
    );
  }

  void _enviarSugerencia(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(Icons.lightbulb_outline, color: Colors.purple[700]),
            const SizedBox(width: 12),
            const Text('Enviar Sugerencia'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Tu opinión es importante para nosotros. Envíanos tus sugerencias para mejorar AMGeCCA.',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.purple[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.purple[200]!),
              ),
              child: Row(
                children: [
                  Icon(Icons.email, color: Colors.purple[700], size: 20),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'amgecca@gmail.com',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: Colors.purple[700],
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.copy, size: 18),
                    onPressed: () =>
                        _copiarAlPortapapeles(context, 'amgecca@gmail.com'),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cerrar'),
          ),
        ],
      ),
    );
  }

  void _copiarAlPortapapeles(BuildContext context, String texto) {
    // TODO: Implementar copiado al portapapeles
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.check_circle, color: Colors.white),
            const SizedBox(width: 12),
            Text('Copiado: $texto'),
          ],
        ),
        backgroundColor: Colors.green[700],
        duration: const Duration(seconds: 2),
      ),
    );
  }
}
