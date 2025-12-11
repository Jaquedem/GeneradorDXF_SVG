import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon
import os

def generar_stl_extruido(ruta_imagen_entrada, ruta_salida_stl, altura_mm=4.0, escala=0.1):
    print(f"Generando modelo 3D desde: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None:
        print("Error: No se carga la imagen.")
        return

    # 1. Preprocesamiento (Tu lógica de limpieza probada)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Contornos con Jerarquía
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("No se detectaron formas.")
        return

    hierarchy = hierarchy[0]
    # hierarchy = [Next, Previous, First_Child, Parent]

    poligonos_shapely = []

    # 3. Reconstrucción de Geometría (Sólidos vs Agujeros)
    # Recorremos solo los contornos que NO tienen padre (Son los bordes exteriores)
    for i, cnt in enumerate(contours):
        
        # Si tiene padre (hierarchy[i][3] != -1), es un contorno interno o basura, 
        # lo procesaremos cuando encontremos a su padre.
        if hierarchy[i][3] != -1:
            continue
            
        # Filtro de ruido para el contorno exterior
        if cv2.contourArea(cnt) < 500:
            continue

        # --- A. Definir el Cascarón (Shell) ---
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Convertir a lista de tuplas (x, -y) para invertir eje vertical y aplicar escala
        shell_coords = [(pt[0][0] * escala, -pt[0][1] * escala) for pt in approx]
        
        if len(shell_coords) < 3: continue

        # --- B. Buscar Agujeros (Holes) ---
        # Buscamos en la lista de contornos aquellos cuyo PADRE sea 'i' (el actual)
        agujeros_coords = []
        
        # Iteramos buscando hijos directos
        # (Nota: Esto es O(N^2) en el peor caso, pero rápido para imágenes simples)
        idx_hijo = hierarchy[i][2] # Primer hijo
        
        while idx_hijo != -1:
            cnt_hijo = contours[idx_hijo]
            
            # Filtro lógico anti-dobles líneas (tu lógica de ratio)
            area_hijo = cv2.contourArea(cnt_hijo)
            area_padre = cv2.contourArea(cnt)
            
            es_agujero_real = True
            
            if area_padre > 0:
                ratio = area_hijo / area_padre
                # Si el hijo es casi del tamaño del padre, es un borde duplicado, NO un agujero.
                if ratio > 0.85: 
                    es_agujero_real = False
            
            if es_agujero_real and area_hijo > 50:
                epsilon_h = 0.001 * cv2.arcLength(cnt_hijo, True)
                approx_h = cv2.approxPolyDP(cnt_hijo, epsilon_h, True)
                h_coords = [(pt[0][0] * escala, -pt[0][1] * escala) for pt in approx_h]
                
                if len(h_coords) >= 3:
                    agujeros_coords.append(h_coords)
            
            # Moverse al siguiente hermano del hijo
            idx_hijo = hierarchy[idx_hijo][0]

        # --- C. Crear Polígono Shapely ---
        # Shapely maneja la matemática de "Sólido menos Agujeros"
        poly = Polygon(shell=shell_coords, holes=agujeros_coords)
        
        # Validar geometría (evita cruces extraños)
        if not poly.is_valid:
            poly = poly.buffer(0) # Truco sucio para arreglar polígonos auto-intersectados
            
        poligonos_shapely.append(poly)

    print(f"Polígonos procesados para extrusión: {len(poligonos_shapely)}")

    if not poligonos_shapely:
        print("No se generaron polígonos válidos.")
        return

    # 4. Extrusión con Trimesh
    mallas = []
    for poly in poligonos_shapely:
        # Extrude polygon crea un objeto Mesh tridimensional
        mesh = trimesh.creation.extrude_polygon(poly, height=altura_mm)
        mallas.append(mesh)

    # Combinar todas las mallas en una sola escena
    mesh_final = trimesh.util.concatenate(mallas)

    # 5. Exportar
    mesh_final.export(ruta_salida_stl)
    print(f"¡Éxito! Archivo STL generado en: {ruta_salida_stl}")
    print(f"Altura de extrusión: {altura_mm}mm")

# --- Ejecución ---
archivo_entrada = r'ChatGPT Image 11 dic 2025, 11_43_56.png'
archivo_salida = 'modelo_3d.stl'

# Ajusta la 'escala' según el tamaño en píxeles de tu imagen.
# Si tu imagen mide 1000px y quieres que mida 100mm, la escala es 0.1
if os.path.exists(archivo_entrada):
    generar_stl_extruido(archivo_entrada, archivo_salida, altura_mm=4.0, escala=0.15)
else:
    print(f"No encuentro el archivo: {archivo_entrada}")

    #C:\Users\jaqueline.tinoco\Documents\generador de contornos\stl.py