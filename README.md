# Análisis de la Red Social a Gran Escala

Este proyecto realiza el análisis y visualización de una red social que contiene información de 10 millones de usuarios. A través de técnicas de procesamiento eficiente, visualización de datos y generación automática de reportes, se extraen métricas clave sobre la estructura y comportamiento de la red.

## Características

- Carga optimizada de archivos de entrada desde Google Drive.
- Procesamiento eficiente con `dask` para archivos grandes.
- Análisis de grafos: número de nodos, aristas, grados y clustering.
- Visualización automática de:
  - Distribución de grados (escala logarítmica).
  - Top 10 usuarios con mayor conectividad.
  - Mapa de densidad de usuarios según ubicación geográfica.
- Decorador para medir el tiempo de ejecución de cada etapa.
- Generación de un reporte automático con estadísticas y tiempos.

## Requisitos

- Python 3.8 o superior
- Paquetes necesarios:
  - pandas
  - dask
  - numpy
  - seaborn
  - matplotlib

Puedes instalar los paquetes con:

```bash
pip install pandas dask numpy seaborn matplotlib
## Datos de Entrada

Los archivos requeridos son:

- `10_million_user.txt.zip`: Archivo con conexiones entre usuarios.
- `10_million_location.txt.zip`: Archivo con coordenadas geográficas de cada usuario.

Ambos archivos deben estar disponibles en tu Google Drive, y el script los copiará y descomprimirá automáticamente.

## Uso

Ejecuta el análisis completo instanciando la clase `RedSocialAnalyzer`:

```python
analyzer = RedSocialAnalyzer()
analyzer.ejecutar_analisis()
