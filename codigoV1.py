#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Proyecto: Análisis y Visualización del Grafo de la Red Social 'X'
Autores: [Tu nombre]
Descripción: Carga masiva y análisis de un grafo de red social con 10M usuarios.
"""

import zipfile
import shutil
import logging
import os
import time
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import dask.dataframe as dd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración avanzada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('red_social_analysis.log'),
        logging.StreamHandler()
    ]
)

def medir_tiempo(func):
    """Decorador para medir tiempos de ejecución."""
    def wrapper(self, *args, **kwargs):
        inicio = time.time()
        resultado = func(self, *args, **kwargs)
        fin = time.time()
        if not hasattr(self, 'tiempos_ejecucion'):
            self.tiempos_ejecucion = {}
        self.tiempos_ejecucion[func.__name__] = fin - inicio
        logging.info(f"⏱️ {func.__name__} ejecutado en {fin - inicio:.2f} segundos")
        return resultado
    return wrapper

class RedSocialAnalyzer:
    """Clase principal para el análisis de la red social."""
    
    def __init__(self):
        self.grafo = None
        self.ubicaciones = None
        self.tiempos_ejecucion = {}
    
    @medir_tiempo
    def cargar_datos_desde_drive(self, ruta_drive: str, archivos: List[str]) -> bool:
        """Copia y descomprime archivos desde Google Drive."""
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            
            for archivo in archivos:
                origen = os.path.join(ruta_drive, archivo)
                destino = f"/content/{archivo}"
                
                if not os.path.exists(origen):
                    logging.error(f" Archivo no encontrado en Drive: {archivo}")
                    return False
                
                shutil.copy(origen, destino)
                logging.info(f" Copiado: {archivo}")
                
                with zipfile.ZipFile(destino, 'r') as zip_ref:
                    zip_ref.extractall("/content")
                logging.info(f" Extraído: {archivo}")
                
            return True
        except Exception as e:
            logging.error(f" Error al cargar desde Drive: {e}")
            return False
    
    @medir_tiempo
    def cargar_ubicaciones_optimizado(self, filepath: str) -> pd.DataFrame:
        """Carga optimizada del archivo de ubicaciones."""
        try:
            if not os.path.exists(filepath):
                logging.error(f" Archivo de ubicaciones no encontrado: {filepath}")
                return pd.DataFrame()
            
            ddf = dd.read_csv(
                filepath, 
                header=None, 
                names=['latitud', 'longitud'],
                dtype={'latitud': 'float32', 'longitud': 'float32'},
                blocksize=25e6
            )
            return ddf.compute()
        except Exception as e:
            logging.error(f"Error al cargar ubicaciones: {e}")
            return pd.DataFrame()
    
    @medir_tiempo
    def cargar_grafo_optimizado(self, filepath: str) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """Carga optimizada del grafo con manejo de formatos inconsistentes."""
        try:
            if not os.path.exists(filepath):
                logging.error(f" Archivo de grafo no encontrado: {filepath}")
                return pd.DataFrame(), {}
            
            conexiones = defaultdict(list)
            usuarios_procesados = 0
            chunk_size = 100000
            
            with open(filepath, 'r') as f:
                while True:
                    lines = f.readlines(chunk_size)
                    if not lines:
                        break
                    
                    for idx, line in enumerate(lines):
                        usuario = usuarios_procesados + idx + 1
                        line = line.strip()
                        if line:
                            try:
                                conexiones[usuario] = [int(x) for x in line.split(',') if x.strip().isdigit()]
                            except Exception as e:
                                logging.warning(f"⚠️ Error procesando usuario {usuario}: {e}")
                    
                    usuarios_procesados += len(lines)
                    if usuarios_procesados % chunk_size == 0:
                        logging.info(f"📊 Procesados {usuarios_procesados} usuarios...")
            
            df = pd.DataFrame({
                'usuario': list(conexiones.keys()), 
                'grado': [len(v) for v in conexiones.values()]
            })
            
            logging.info(f"✅ Grafo cargado con {len(df)} nodos y {df['grado'].sum()} aristas")
            return df, conexiones
        except Exception as e:
            logging.error(f"❌ Error al cargar el grafo: {e}")
            return pd.DataFrame(), {}
    
    @medir_tiempo
    def calcular_estadisticas_avanzadas(self, df: pd.DataFrame, conexiones: Dict[int, List[int]]) -> Dict:
        """Calcula estadísticas avanzadas del grafo."""
        stats = {}
        
        if not df.empty:
            stats['n_nodos'] = df.shape[0]
            stats['n_aristas'] = df['grado'].sum() // 2
            stats['grado_promedio'] = df['grado'].mean()
            stats['grado_max'] = df['grado'].max()
            stats['grado_min'] = df['grado'].min()
            
            stats['distribucion_grados'] = df['grado'].value_counts().sort_index()
            
            if len(conexiones) > 0:
                sample_size = min(1000, len(conexiones))
                sample_nodes = np.random.choice(
                    list(conexiones.keys()), 
                    size=sample_size, 
                    replace=False
                )
                
                clustering_total = 0
                nodos_validos = 0
                
                for nodo in sample_nodes:
                    try:
                        vecinos = set(conexiones[nodo])
                        n_vecinos = len(vecinos)
                        if n_vecinos < 2:
                            continue
                        
                        enlaces = 0
                        for v in vecinos:
                            if v in conexiones:
                                enlaces += len(vecinos & set(conexiones[v]))
                        
                        clustering_total += enlaces / (n_vecinos * (n_vecinos - 1))
                        nodos_validos += 1
                    except Exception as e:
                        logging.warning(f"⚠️ Error calculando clustering para nodo {nodo}: {e}")
                
                stats['clustering_promedio'] = clustering_total / nodos_validos if nodos_validos > 0 else 0.0
        
        return stats
    
    def generar_visualizaciones(self, df: pd.DataFrame, ubicaciones: pd.DataFrame, stats: dict):
        """Genera 3 gráficos clave con muestreo optimizado"""
        try:
            plt.figure(figsize=(18, 6))
            
            # Gráfico 1: Distribución de grados (log-log)
            plt.subplot(1, 3, 1)
            sns.histplot(df['grado'], bins=100, kde=False, log_scale=(True, True))
            plt.title('Distribución de Grados (Log-Log)')
            plt.xlabel('Grado')
            plt.ylabel('Frecuencia')
            
            # Gráfico 2: Top 10 usuarios más conectados
            plt.subplot(1, 3, 2)
            top_users = df.nlargest(10, 'grado')
            sns.barplot(x='grado', y='usuario', data=top_users, orient='h', palette='viridis')
            plt.title('Top 10 Usuarios más Conectados')
            plt.xlabel('Conexiones')
            plt.ylabel('ID Usuario')
            
            # Gráfico 3: Mapa de densidad geográfica
            plt.subplot(1, 3, 3)
            muestra = ubicaciones.sample(10000)  # Muestra del 0.1%
            sns.kdeplot(x=muestra['longitud'], y=muestra['latitud'], 
                       cmap='viridis', fill=True, thresh=0, levels=50)
            plt.title('Densidad Geográfica (Muestra 0.1%)')
            plt.xlabel('Longitud')
            plt.ylabel('Latitud')
            
            plt.tight_layout()
            plt.savefig('analisis_red.png', dpi=100)
            plt.show()
            
        except Exception as e:
            logging.error(f"Error en visualizaciones: {str(e)}")
    
    def generar_reporte(self, stats: Dict):
        """Genera un reporte completo del análisis."""
        try:
            with open("reporte_analisis.txt", "w") as f:
                f.write("=== REPORTE DE ANÁLISIS DE RED SOCIAL ===\n\n")
                f.write(" ESTADÍSTICAS PRINCIPALES:\n")
                for key, value in stats.items():
                    if key != 'distribucion_grados':
                        f.write(f"- {key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\n⏱ TIEMPOS DE EJECUCIÓN:\n")
                for func, tiempo in self.tiempos_ejecucion.items():
                    f.write(f"- {func}: {tiempo:.2f} segundos\n")
                
                f.write("\n RECOMENDACIONES:\n")
                if stats.get('grado_promedio', 0) < 10:
                    f.write("- La red tiene baja conectividad promedio\n")
                if stats.get('clustering_promedio', 0) > 0.5:
                    f.write("- Alta clusterización: existen grupos muy conectados internamente\n")
                
            logging.info(" Reporte generado: reporte_analisis.txt")
        except Exception as e:
            logging.error(f" Error generando reporte: {e}")
    
    def ejecutar_analisis(self):
        """Método principal para ejecutar todo el análisis."""
        try:
            logging.info(" Iniciando análisis de red social...")
            
            ruta_drive = "/content/drive/MyDrive/Colab Notebooks/"
            archivos = ["10_million_user.txt.zip", "10_million_location.txt.zip"]
            
            if not self.cargar_datos_desde_drive(ruta_drive, archivos):
                return False
            
            ubicacion_file = "/content/10_million_location.txt"
            usuario_file = "/content/10_million_user.txt"
            
            self.ubicaciones = self.cargar_ubicaciones_optimizado(ubicacion_file)
            grafo_df, conexiones = self.cargar_grafo_optimizado(usuario_file)
            
            if grafo_df.empty:
                logging.error(" No se pudo cargar el grafo.")
                return False
                
            stats = self.calcular_estadisticas_avanzadas(grafo_df, conexiones)
            
            # Generar y mostrar visualizaciones
            self.generar_visualizaciones(grafo_df, self.ubicaciones, stats)
            
            print("\n ESTADÍSTICAS AVANZADAS DEL GRAFO:")
            for key, value in stats.items():
                if key != 'distribucion_grados':
                    print(f"🔹 {key.replace('_', ' ').title()}: {value}")
            
            self.generar_reporte(stats)
            
            print("\n⏱ TIEMPOS DE EJECUCIÓN:")
            for func, tiempo in self.tiempos_ejecucion.items():
                print(f"• {func}: {tiempo:.2f} segundos")
            
            return True
            
        except Exception as e:
            logging.error(f" Error en el análisis: {e}")
            return False

if __name__ == "__main__":
    analyzer = RedSocialAnalyzer()
    if analyzer.ejecutar_analisis():
        logging.info(" Análisis completado exitosamente!")
    else
        logging.error("El análisis encontró problemas.")