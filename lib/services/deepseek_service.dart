// lib/services/deepseek_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class DeepSeekService {
  final String apiKey = "sk-a6314e8738454ad893a4c50bcaacf242"; //API KEY

  Future<String> obtenerRecomendacionesClima({
    required double temperatura,
    required double humedad,
    required double viento,
    required String descripcionClima,
    required bool esAptoParaPulverizar,
  }) async {
    try {
      final prompt =
          """
Soy un asistente agrícola. Analiza estas condiciones climáticas actuales:

📊 Condiciones:
- Temperatura: ${temperatura.toStringAsFixed(1)}°C
- Humedad relativa: ${humedad.toStringAsFixed(0)}%
- Velocidad del viento: ${viento.toStringAsFixed(1)} km/h
- Clima: $descripcionClima
- Estado para pulverización: ${esAptoParaPulverizar ? 'APTO ✅' : 'NO APTO ⚠️'}

Proporciona recomendaciones breves y específicas para:
1. ¿Es buen momento para aplicar pesticidas/fertilizantes?
2. ¿Qué precauciones tomar?
3. ¿Cuál es el mejor momento del día para trabajar?

Responde en máximo 150 palabras, de forma práctica y directa para un agricultor.
""";

      final url = Uri.parse("https://api.deepseek.com/chat/completions");

      final response = await http.post(
        url,
        headers: {
          "Authorization": "Bearer $apiKey",
          "Content-Type": "application/json",
        },
        body: json.encode({
          "model": "deepseek-chat",
          "messages": [
            {"role": "user", "content": prompt},
          ],
          "temperature": 0.7,
          "max_tokens": 300,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data["choices"] != null &&
            data["choices"].isNotEmpty &&
            data["choices"][0]["message"] != null) {
          return data["choices"][0]["message"]["content"] ?? "Sin respuesta";
        }
      } else if (response.statusCode == 402) {
        return "⚠️ Error: Saldo insuficiente en tu cuenta de DeepSeek.";
      } else if (response.statusCode == 401) {
        return "⚠️ Error: API key inválida. Verifica tu clave.";
      }

      return "❌ Error al obtener recomendaciones de IA";
    } catch (e) {
      print("❌ Error DeepSeek: $e");
      return "❌ Error de conexión con el servicio de IA";
    }
  }
}
