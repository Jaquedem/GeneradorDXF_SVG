import cv2
import ezdxf
import numpy as np
import os

def imagen_a_dxf(ruta_imagen_entrada, ruta_salida_dxf):
    print(f"Iniciando conversión a DXF: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None:
        print("Error: No se encuentra la imagen.")
        return

    # 1. Procesamiento de imagen (Usando Canny para bordes finos)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ajuste de Canny: 
    # Umbrales 50-150 son un buen punto de partida para detectar bordes estructurales
    edges = cv2.Canny(gray, 50, 150)

    # 2. Encontrar contornos
    # RETR_LIST: Obtiene todos los contornos sin jerarquía (más simple para DXF plano)
    # CHAIN_APPROX_NONE: Sin compresión, máxima fidelidad de puntos
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print(f"Entidades detectadas: {len(contours)}")

    # 3. Configurar documento DXF (Versión R2010 es muy compatible)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace() # Aquí es donde se "dibuja"

    # Convertir contornos a Polilíneas DXF
    contador_entidades = 0
    
    for cnt in contours:
        # Filtro de ruido: ignorar motas de polvo (menores a 15px de longitud)
        if cv2.arcLength(cnt, True) < 15:
            continue

        # Suavizado ligero para que la máquina no "vibre" con miles de micropuntos
        # epsilon bajo = alta precisión. 
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Preparar puntos para ezdxf
        # IMPORTANTE: Invertimos Y (-pt[0][1]) para que el dibujo no salga de cabeza
        # debido a la diferencia de coordenadas entre Imágenes (Top-Left) y CAD (Bottom-Left)
        puntos_dxf = [(float(pt[0][0]), -float(pt[0][1])) for pt in approx]

        # Añadir LWPOLYLINE (Lightweight Polyline)
        # Es la entidad más eficiente para contornos 2D
        msp.add_lwpolyline(puntos_dxf, close=True, dxfattribs={'layer': 'CORTE', 'color': 1})
        contador_entidades += 1

    # Guardar
    doc.saveas(ruta_salida_dxf)
    print(f"¡Éxito! Se generaron {contador_entidades} polilíneas en: {ruta_salida_dxf}")

# --- Ejecución ---
archivo_entrada = r'ChatGPT Image 11 dic 2025, 11_29_31.png'
archivo_salida = 'diseño_corte2.dxf'

if os.path.exists(archivo_entrada):
    imagen_a_dxf(archivo_entrada, archivo_salida)
else:
    print(f"No encuentro el archivo: {archivo_entrada}")