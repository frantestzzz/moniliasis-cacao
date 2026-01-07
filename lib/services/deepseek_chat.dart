import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class DeepSeekChat {
  final String apiKey = "sk-a6314e8738454ad893a4c50bcaacf242";
  final String baseUrl = "https://api.deepseek.com";

  // Enviar mensaje de texto simple
  Future<String> sendMessage(String message) async {
    try {
      final url = Uri.parse("$baseUrl/chat/completions");

      final response = await http.post(
        url,
        headers: {
          "Authorization": "Bearer $apiKey",
          "Content-Type": "application/json",
        },
        body: json.encode({
          "model": "deepseek-chat",
          "messages": [
            {"role": "user", "content": message},
          ],
          "temperature": 0.7,
        }),
      );

      print("📤 Status Code: ${response.statusCode}");
      return _handleResponse(response);
    } catch (e) {
      print("❌ Exception: $e");
      return "❌ Error de conexión: $e";
    }
  }

  // ⚠️ IMPORTANTE: DeepSeek requiere procesamiento especial para imágenes
  // El modelo deepseek-chat NO soporta imágenes directamente
  // Esta función usa un enfoque alternativo
  Future<String> sendMessageWithImage(String message, File imageFile) async {
    try {
      // Opción 1: Indicar al usuario que el modelo no soporta imágenes
      return """
⚠️ AVISO IMPORTANTE:

El modelo DeepSeek Chat actual no tiene capacidad de visión por computadora.

Para analizar imágenes de cultivos, puedes:

1. 📝 Describir la imagen con texto:
   - Color de las hojas
   - Manchas o decoloración
   - Forma de las lesiones
   - Ubicación del daño

2. 🔍 Usar el sistema de detección local (el botón de cámara/galería en la pantalla principal)

3. 💡 Actualizar a un modelo con visión (si DeepSeek lanza uno)

¿Quieres que te ayude a interpretar tu descripción de la planta?
""";

      // Opción 2: Si DeepSeek lanza un modelo con visión, descomenta esto:
      /*
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);

      final url = Uri.parse("$baseUrl/v1/chat/completions");

      final response = await http.post(
        url,
        headers: {
          "Authorization": "Bearer $apiKey",
          "Content-Type": "application/json",
        },
        body: json.encode({
          "model": "deepseek-vl", // Modelo con visión (cuando esté disponible)
          "messages": [
            {
              "role": "user",
              "content": [
                {"type": "text", "text": message},
                {
                  "type": "image_url",
                  "image_url": {
                    "url": "data:image/jpeg;base64,$base64Image"
                  }
                }
              ]
            }
          ],
          "temperature": 0.7,
        }),
      );

      print("📸 Status Code con imagen: ${response.statusCode}");
      print("📸 Response: ${response.body}");
      
      return _handleResponse(response);
      */
    } catch (e) {
      print("❌ Exception con imagen: $e");
      return "❌ Error al procesar imagen: $e";
    }
  }

  // Enviar mensaje con documento
  Future<String> sendMessageWithDocument(
    String message,
    File docFile,
    String fileName,
  ) async {
    try {
      String fileContent = "";

      // Leer contenido según el tipo de archivo
      if (fileName.toLowerCase().endsWith('.txt')) {
        fileContent = await docFile.readAsString();
        fileContent = fileContent.substring(
          0,
          fileContent.length > 10000 ? 10000 : fileContent.length,
        ); // Limitar a 10k caracteres
      } else if (fileName.toLowerCase().endsWith('.csv')) {
        fileContent = await docFile.readAsString();
        final lines = fileContent.split('\n').take(50).join('\n');
        fileContent = "Primeras 50 líneas del CSV:\n$lines";
      } else {
        final bytes = await docFile.readAsBytes();
        fileContent =
            """
Archivo adjunto: $fileName
Tamaño: ${(bytes.length / 1024).toStringAsFixed(2)} KB
Tipo: ${fileName.split('.').last.toUpperCase()}

⚠️ Este tipo de archivo no puede ser leído directamente.
Por favor, describe su contenido o comparte un archivo .txt o .csv
""";
      }

      final url = Uri.parse("$baseUrl/chat/completions");

      final response = await http.post(
        url,
        headers: {
          "Authorization": "Bearer $apiKey",
          "Content-Type": "application/json",
        },
        body: json.encode({
          "model": "deepseek-chat",
          "messages": [
            {
              "role": "user",
              "content":
                  """
$message

--- CONTENIDO DEL DOCUMENTO ---
Archivo: $fileName

$fileContent
""",
            },
          ],
          "temperature": 0.7,
          "max_tokens": 2000,
        }),
      );

      print("📄 Status Code con documento: ${response.statusCode}");
      return _handleResponse(response);
    } catch (e) {
      print("❌ Exception con documento: $e");
      return "❌ Error al procesar documento: $e";
    }
  }

  String _handleResponse(http.Response response) {
    try {
      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        if (data["choices"] != null &&
            data["choices"].isNotEmpty &&
            data["choices"][0]["message"] != null) {
          return data["choices"][0]["message"]["content"] ?? "Sin respuesta";
        } else {
          return "⚠️ Respuesta vacía del servidor";
        }
      } else if (response.statusCode == 400) {
        final errorData = json.decode(response.body);
        final errorMsg =
            errorData["error"]?["message"] ??
            errorData["message"] ??
            "Solicitud inválida";

        // Mensajes específicos para errores comunes
        if (errorMsg.contains("image") || errorMsg.contains("vision")) {
          return """
⚠️ Este modelo no soporta análisis de imágenes.

Usa el sistema de detección local (botón cámara/galería en la pantalla principal) para analizar fotos de cultivos.

O describe la imagen con texto y te ayudaré.
""";
        }

        return "⚠️ Error 400: $errorMsg";
      } else if (response.statusCode == 401) {
        return "🔑 Error: API key inválida o expirada. Verifica tu clave en platform.deepseek.com";
      } else if (response.statusCode == 402) {
        return "💳 Sin saldo: Recarga tu cuenta en platform.deepseek.com";
      } else if (response.statusCode == 429) {
        return "⏱️ Demasiadas solicitudes. Espera unos segundos e intenta nuevamente.";
      } else if (response.statusCode == 503) {
        return "🔧 Servicio temporalmente no disponible. Intenta en unos minutos.";
      } else {
        final errorData = json.decode(response.body);
        final errorMsg =
            errorData["error"]?["message"] ??
            errorData["message"] ??
            response.body;
        return "❌ Error ${response.statusCode}: $errorMsg";
      }
    } catch (e) {
      return "❌ Error procesando respuesta: $e\n\nRespuesta del servidor: ${response.body}";
    }
  }
}
