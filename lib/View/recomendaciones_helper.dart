import 'package:flutter/material.dart';

class Recomendacion {
  final String enfermedad;
  final String descripcion;
  final List<String> sintomas;
  final List<String> tratamientos;
  final List<String> prevencion;
  final IconData icono;
  final Color color;

  const Recomendacion({
    required this.enfermedad,
    required this.descripcion,
    required this.sintomas,
    required this.tratamientos,
    required this.prevencion,
    required this.icono,
    required this.color,
  });
}

class RecomendacionesHelper {
  // Base de datos de recomendaciones
  static final Map<String, Recomendacion> _recomendaciones = {
    // ☕ CAFÉ
    'Cafe_Roya_de_la_hoja': Recomendacion(
      enfermedad: 'Roya del Café',
      descripcion:
          'Enfermedad fungosa causada por Hemileia vastatrix que afecta las hojas.',
      sintomas: [
        'Manchas amarillas en el envés de las hojas',
        'Polvo anaranjado (esporas)',
        'Defoliación prematura',
      ],
      tratamientos: [
        'Fungicidas cúpricos: Oxicloruro de cobre (3-4 g/L)',
        'Fungicidas sistémicos: Triadimefon o Propiconazol',
        'Aplicar cada 21 días en época lluviosa',
      ],
      prevencion: [
        'Podar ramas enfermas',
        'Mejorar ventilación entre plantas',
        'Fertilizar adecuadamente',
      ],
      icono: Icons.coronavirus,
      color: Colors.orange,
    ),
    'Cafe_Cercospora': Recomendacion(
      enfermedad: 'Mancha de Cercospora',
      descripcion: 'Hongo Cercospora coffeicola que causa manchas en hojas.',
      sintomas: [
        'Manchas circulares marrones con halo amarillo',
        'Centro gris claro',
        'Caída de hojas',
      ],
      tratamientos: [
        'Fungicidas: Mancozeb (2 g/L)',
        'Clorotalonil (2 ml/L)',
        'Aplicar cada 15-20 días',
      ],
      prevencion: [
        'Reducir sombra excesiva',
        'Evitar humedad prolongada',
        'Mantener nutrición balanceada',
      ],
      icono: Icons.bug_report,
      color: Colors.brown,
    ),
    'Cafe_Minero': Recomendacion(
      enfermedad: 'Minador de la Hoja',
      descripcion: 'Larva de Leucoptera coffeella que crea túneles en hojas.',
      sintomas: [
        'Túneles o minas en las hojas',
        'Manchas necróticas',
        'Reducción de fotosíntesis',
      ],
      tratamientos: [
        'Insecticidas: Thiametoxam',
        'Abamectina (0.5 ml/L)',
        'Control biológico: Parasitoides',
      ],
      prevencion: [
        'Eliminar hojas afectadas',
        'Regular sombrío',
        'Monitoreo constante',
      ],
      icono: Icons.pest_control,
      color: Colors.red,
    ),
    'Cafe_Phoma': Recomendacion(
      enfermedad: 'Phoma',
      descripcion: 'Hongo que causa manchas foliares y deterioro general.',
      sintomas: [
        'Manchas irregulares café oscuro',
        'Puntos negros (picnidios)',
        'Defoliación',
      ],
      tratamientos: [
        'Fungicidas: Carbendazim',
        'Azoxistrobina + Difenoconazol',
        'Aplicar preventivamente',
      ],
      prevencion: [
        'Podar y quemar material enfermo',
        'Mejorar drenaje',
        'Evitar heridas en plantas',
      ],
      icono: Icons.water_drop,
      color: Colors.blueGrey,
    ),
    'Cafe_Arana_roja': Recomendacion(
      enfermedad: 'Araña Roja',
      descripcion: 'Ácaro Oligonychus yothersi que se alimenta de hojas.',
      sintomas: [
        'Puntos amarillos en hojas',
        'Telarañas finas',
        'Hojas bronceadas y secas',
      ],
      tratamientos: [
        'Acaricidas: Abamectina (0.5 ml/L)',
        'Azufre mojable (3 g/L)',
        'Aceite agrícola al 1%',
      ],
      prevencion: [
        'Mantener humedad adecuada',
        'Evitar estrés hídrico',
        'Control biológico con depredadores',
      ],
      icono: Icons.bug_report,
      color: Colors.red[900]!,
    ),

    // 🍫 CACAO
    'Cacao_Podredumbre_negra': Recomendacion(
      enfermedad: 'Podredumbre Negra',
      descripcion: 'Causada por Phytophthora palmivora, afecta mazorcas.',
      sintomas: [
        'Manchas oscuras en mazorcas',
        'Pudrición interna',
        'Pérdida total del fruto',
      ],
      tratamientos: [
        'Fungicidas cúpricos: Hidróxido de cobre (3 g/L)',
        'Metalaxyl + Mancozeb',
        'Fosetyl-Al (2.5 g/L)',
      ],
      prevencion: [
        'Podar ramas bajas',
        'Mejorar drenaje del suelo',
        'Cosechar frecuentemente',
      ],
      icono: Icons.coronavirus,
      color: Colors.black87,
    ),
    'Cacao_Barrenador': Recomendacion(
      enfermedad: 'Barrenador del Fruto',
      descripcion: 'Larvas de insectos que perforan las mazorcas.',
      sintomas: [
        'Orificios en la cáscara',
        'Galerías internas',
        'Fermentación prematura',
      ],
      tratamientos: [
        'Insecticidas: Clorpirifos',
        'Lambda-cialotrina (0.5 ml/L)',
        'Control cultural: eliminación de frutos',
      ],
      prevencion: [
        'Cosechar regularmente',
        'Eliminar frutos enfermos',
        'Mantener limpieza del cultivo',
      ],
      icono: Icons.pest_control,
      color: Colors.brown[800]!,
    ),

    // 🌾 ARROZ
    'Arroz_Tizon_bacteriano': Recomendacion(
      enfermedad: 'Tizón Bacteriano',
      descripcion: 'Bacteria Xanthomonas oryzae que causa marchitez.',
      sintomas: [
        'Rayas amarillas en hojas',
        'Marchitez de plantas jóvenes',
        'Muerte de plántulas',
      ],
      tratamientos: [
        'Antibióticos: Estreptomicina (0.2 g/L)',
        'Oxitetraciclina',
        'Cobre bactericida (2 g/L)',
      ],
      prevencion: [
        'Usar semilla certificada',
        'Variedades resistentes',
        'Control de malezas hospederas',
      ],
      icono: Icons.warning,
      color: Colors.yellow[800]!,
    ),
    'Arroz_Tizon_de_la_hoja': Recomendacion(
      enfermedad: 'Tizón de la Hoja (Blast)',
      descripcion: 'Hongo Pyricularia oryzae, enfermedad devastadora.',
      sintomas: [
        'Lesiones con forma de diamante',
        'Centro gris con borde café',
        'Muerte de panojas',
      ],
      tratamientos: [
        'Fungicidas: Triciclazol (0.6 g/L)',
        'Azoxistrobina + Tebuconazol',
        'Aplicar al inicio de síntomas',
      ],
      prevencion: [
        'Rotación de cultivos',
        'Fertilización balanceada',
        'Variedades resistentes',
      ],
      icono: Icons.coronavirus,
      color: Colors.red[700]!,
    ),
    'Arroz_Mancha_marron': Recomendacion(
      enfermedad: 'Mancha Marrón',
      descripcion: 'Hongo Bipolaris oryzae en condiciones de estrés.',
      sintomas: [
        'Manchas ovaladas marrones',
        'Centro gris claro',
        'Afecta hojas y granos',
      ],
      tratamientos: [
        'Fungicidas: Mancozeb (2 g/L)',
        'Propiconazol',
        'Validamicina',
      ],
      prevencion: [
        'Mejorar nutrición (potasio)',
        'Evitar exceso de nitrógeno',
        'Buen manejo del agua',
      ],
      icono: Icons.circle,
      color: Colors.brown,
    ),
    'Arroz_Escaldadura_de_la_hoja': Recomendacion(
      enfermedad: 'Escaldadura de la Hoja',
      descripcion: 'Bacteria que causa quemaduras en las hojas.',
      sintomas: [
        'Manchas alargadas color paja',
        'Bordes ondulados',
        'Afecta producción de granos',
      ],
      tratamientos: [
        'Fungicidas: Validamicina (2 ml/L)',
        'Kasugamicina',
        'Aplicar preventivamente',
      ],
      prevencion: [
        'Manejo adecuado del agua',
        'Variedades tolerantes',
        'Eliminar rastrojos',
      ],
      icono: Icons.local_fire_department,
      color: Colors.orange[700]!,
    ),
    'Arroz_Tizon_de_la_vaina': Recomendacion(
      enfermedad: 'Tizón de la Vaina',
      descripcion: 'Hongo Rhizoctonia solani que ataca las vainas.',
      sintomas: [
        'Lesiones elípticas en vainas',
        'Pudrición del tallo',
        'Vaneamiento de granos',
      ],
      tratamientos: [
        'Fungicidas: Validamicina (2 ml/L)',
        'Hexaconazol',
        'Pencicurón',
      ],
      prevencion: [
        'Evitar siembras densas',
        'Reducir humedad relativa',
        'Fertilización balanceada',
      ],
      icono: Icons.grass,
      color: Colors.green[900]!,
    ),

    // 🌽 MAÍZ
    'Maiz_Mancha_gris': Recomendacion(
      enfermedad: 'Mancha Gris',
      descripcion: 'Hongo Cercospora zeae-maydis en hojas.',
      sintomas: [
        'Manchas rectangulares grises',
        'Lesiones paralelas a nervaduras',
        'Reducción de área foliar',
      ],
      tratamientos: [
        'Fungicidas: Azoxistrobina (0.8 ml/L)',
        'Triazoles + Estrobilurinas',
        'Aplicar en etapas críticas',
      ],
      prevencion: [
        'Rotación de cultivos',
        'Híbridos resistentes',
        'Eliminar rastrojos',
      ],
      icono: Icons.view_module,
      color: Colors.grey,
    ),
    'Maiz_Roya_comun': Recomendacion(
      enfermedad: 'Roya Común',
      descripcion: 'Hongo Puccinia sorghi con pústulas color óxido.',
      sintomas: [
        'Pústulas circulares marrones',
        'Polvo rojizo (esporas)',
        'Amarillamiento de hojas',
      ],
      tratamientos: [
        'Fungicidas: Tebuconazol (1 ml/L)',
        'Azoxistrobina + Ciproconazol',
        'Aplicar preventivamente',
      ],
      prevencion: [
        'Sembrar híbridos resistentes',
        'Fechas de siembra adecuadas',
        'Eliminar plantas voluntarias',
      ],
      icono: Icons.coronavirus,
      color: Colors.orange[900]!,
    ),
    'Maiz_Tizon': Recomendacion(
      enfermedad: 'Tizón del Norte',
      descripcion: 'Exserohilum turcicum causa lesiones alargadas.',
      sintomas: [
        'Lesiones largas gris-verdosas',
        'Forma elíptica o fusiforme',
        'Muerte prematura de hojas',
      ],
      tratamientos: [
        'Fungicidas: Mancozeb + Metalaxyl',
        'Propiconazol (1 ml/L)',
        'Inicio al ver primeros síntomas',
      ],
      prevencion: [
        'Variedades tolerantes',
        'Rotación con otros cultivos',
        'Manejo de rastrojos',
      ],
      icono: Icons.arrow_upward,
      color: Colors.blueGrey[800]!,
    ),

    // 🍅 TOMATE (ejemplos, completa según tu modelo)
    'Tomate_Tizon_Temprano': Recomendacion(
      enfermedad: 'Tizón Temprano',
      descripcion: 'Alternaria solani causa manchas concéntricas.',
      sintomas: [
        'Manchas marrones con anillos',
        'Afecta hojas inferiores primero',
        'Defoliación progresiva',
      ],
      tratamientos: [
        'Fungicidas: Mancozeb (2 g/L)',
        'Clorotalonil',
        'Azoxistrobina',
      ],
      prevencion: [
        'Rotación de cultivos',
        'Riego por goteo',
        'Eliminar hojas infectadas',
      ],
      icono: Icons.coronavirus,
      color: Colors.brown,
    ),

    // 🍌 PLÁTANO
    'Platano_Sigatoka': Recomendacion(
      enfermedad: 'Sigatoka Negra',
      descripcion:
          'Mycosphaerella fijiensis, enfermedad foliar grave que afecta la fotosíntesis.',
      sintomas: [
        'Rayas amarillas que se vuelven negras',
        'Necrosis de hojas',
        'Reducción de producción hasta 50%',
        'Manchas alargadas paralelas a las nervaduras',
      ],
      tratamientos: [
        'Fungicidas: Mancozeb (3 g/L)',
        'Propiconazol + Azoxistrobina (1 ml/L)',
        'Aceite mineral (10 ml/L)',
        'Alternar productos para evitar resistencia',
      ],
      prevencion: [
        'Deshoje sanitario semanal',
        'Mejorar drenaje del suelo',
        'Variedades resistentes',
        'Eliminar hojas con más de 50% de daño',
      ],
      icono: Icons.coronavirus,
      color: Colors.black,
    ),
    'Platano_Cordana': Recomendacion(
      enfermedad: 'Cordana',
      descripcion:
          'Hongo Cordana musae que causa manchas foliares y pudrición del fruto.',
      sintomas: [
        'Manchas circulares marrones en hojas',
        'Centro gris con halo amarillo',
        'Pudrición en corona del racimo',
        'Manchas negras en dedos del racimo',
      ],
      tratamientos: [
        'Fungicidas: Mancozeb (2.5 g/L)',
        'Clorotalonil (2 ml/L)',
        'Azoxistrobina (0.8 ml/L)',
        'Aplicar cada 14-21 días',
      ],
      prevencion: [
        'Eliminar hojas infectadas',
        'Evitar heridas en plantas',
        'Mejorar ventilación del cultivo',
        'Desinfectar herramientas de corte',
      ],
      icono: Icons.circle,
      color: Colors.brown[700]!,
    ),
    'Platano_Pestalotiopsis': Recomendacion(
      enfermedad: 'Pestalotiopsis',
      descripcion:
          'Hongo Pestalotiopsis que causa manchas foliares y necrosis en puntas.',
      sintomas: [
        'Manchas irregulares color café',
        'Necrosis en márgenes y puntas de hojas',
        'Puntos negros (acérvulos) en lesiones',
        'Secamiento progresivo del follaje',
      ],
      tratamientos: [
        'Fungicidas: Carbendazim (1.5 g/L)',
        'Benomyl (1 g/L)',
        'Mancozeb + Metalaxyl',
        'Aplicaciones preventivas cada 21 días',
      ],
      prevencion: [
        'Eliminar tejido necrótico',
        'Mantener nutrición balanceada',
        'Evitar estrés hídrico',
        'Control de otros patógenos',
      ],
      icono: Icons.warning,
      color: Colors.amber[900]!,
    ),
    'Platano_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu plátano está en excelentes condiciones.',
      sintomas: [
        'Hojas grandes y verdes',
        'Racimos desarrollándose bien',
        'Sin manchas ni deformaciones',
        'Crecimiento vigoroso',
      ],
      tratamientos: [
        'Mantener fertilización NPK balanceada',
        'Deshije oportuno (dejar hijo espada)',
        'Monitoreo semanal de plagas',
        'Riego según necesidad',
      ],
      prevencion: [
        'Control de malezas',
        'Drenaje adecuado',
        'Inspecciones frecuentes',
        'Rotación de hijuelos',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
    'Cafe_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu planta de café está en buen estado.',
      sintomas: [
        'Hojas verdes brillantes',
        'Crecimiento vigoroso',
        'Sin manchas ni deformaciones',
      ],
      tratamientos: [
        'Mantener programa de fertilización',
        'Riego adecuado',
        'Monitoreo preventivo regular',
      ],
      prevencion: [
        'Continuar prácticas culturales',
        'Inspecciones semanales',
        'Nutrición balanceada',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
    'Cacao_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu planta de cacao está en excelente estado.',
      sintomas: [
        'Follaje abundante y verde',
        'Mazorcas sanas',
        'Buen desarrollo',
      ],
      tratamientos: [
        'Mantener fertilización',
        'Continuar monitoreo',
        'Podas de mantenimiento',
      ],
      prevencion: [
        'Seguir calendario de manejo',
        'Control preventivo de plagas',
        'Manejo de sombra adecuado',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
    'Arroz_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu cultivo de arroz está en óptimas condiciones.',
      sintomas: [
        'Plantas vigorosas',
        'Color verde intenso',
        'Buen macollamiento',
      ],
      tratamientos: [
        'Continuar fertilización programada',
        'Manejo apropiado del agua',
        'Monitoreo regular',
      ],
      prevencion: [
        'Mantener nivel de agua óptimo',
        'Control preventivo',
        'Fertilización oportuna',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
    'Maiz_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu maíz está creciendo correctamente.',
      sintomas: [
        'Hojas verdes y erectas',
        'Buen desarrollo de mazorcas',
        'Sin estrés visible',
      ],
      tratamientos: [
        'Mantener programa nutricional',
        'Riego según necesidad',
        'Monitoreo de plagas',
      ],
      prevencion: [
        'Control de malezas',
        'Fertilización adecuada',
        'Inspecciones regulares',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
    'Tomate_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu planta de tomate está perfecta.',
      sintomas: [
        'Follaje abundante',
        'Flores y frutos sanos',
        'Crecimiento normal',
      ],
      tratamientos: [
        'Continuar fertilización',
        'Riego constante',
        'Tutorado apropiado',
      ],
      prevencion: [
        'Poda de mantenimiento',
        'Control preventivo',
        'Ventilación adecuada',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
    'Platano_Saludable': Recomendacion(
      enfermedad: '¡Planta Saludable!',
      descripcion: 'Tu plátano está en excelentes condiciones.',
      sintomas: [
        'Hojas grandes y verdes',
        'Racimos desarrollándose bien',
        'Sin daños visibles',
      ],
      tratamientos: [
        'Mantener fertilización',
        'Deshije oportuno',
        'Monitoreo regular',
      ],
      prevencion: [
        'Control de malezas',
        'Drenaje adecuado',
        'Inspecciones frecuentes',
      ],
      icono: Icons.check_circle,
      color: Colors.green,
    ),
  };

  static Recomendacion? obtenerRecomendacion(String enfermedad) {
    // Normalizar el nombre de la enfermedad
    String normalizado = enfermedad
        .replaceAll(' ', '_')
        .replaceAll('á', 'a')
        .replaceAll('é', 'e')
        .replaceAll('í', 'i')
        .replaceAll('ó', 'o')
        .replaceAll('ú', 'u')
        .replaceAll('ñ', 'n');

    return _recomendaciones[normalizado];
  }
}
