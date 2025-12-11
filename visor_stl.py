import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon
import os

def generar_stl_con_visor(ruta_imagen_entrada, ruta_salida_stl, altura_mm=4.0, escala=0.15):
    print(f"Procesando modelo 3D: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None:
        print("Error: No se carga la imagen.")
        return

    # 1. Preprocesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Contornos con Jerarquía
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("No se detectaron formas.")
        return

    hierarchy = hierarchy[0]
    poligonos_shapely = []

    # 3. Reconstrucción de Geometría (Sólidos - Agujeros)
    for i, cnt in enumerate(contours):
        
        # Procesamos solo contornos externos (que serán el cuerpo sólido)
        # hierarchy format: [Next, Previous, First_Child, Parent]
        if hierarchy[i][3] != -1:
            continue
            
        if cv2.contourArea(cnt) < 500:
            continue

        # --- Definir el Cascarón (Shell) ---
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Aplicamos escala e invertimos Y
        shell_coords = [(pt[0][0] * escala, -pt[0][1] * escala) for pt in approx]
        
        if len(shell_coords) < 3: continue

        # --- Buscar Agujeros (Holes) ---
        agujeros_coords = []
        idx_hijo = hierarchy[i][2] # Primer hijo
        
        while idx_hijo != -1:
            cnt_hijo = contours[idx_hijo]
            
            # Tu lógica de filtro para evitar dobles líneas
            area_hijo = cv2.contourArea(cnt_hijo)
            area_padre = cv2.contourArea(cnt)
            
            es_agujero_real = True
            if area_padre > 0:
                ratio = area_hijo / area_padre
                if ratio > 0.85: # Si es casi igual al padre, es ruido
                    es_agujero_real = False
            
            if es_agujero_real and area_hijo > 50:
                epsilon_h = 0.001 * cv2.arcLength(cnt_hijo, True)
                approx_h = cv2.approxPolyDP(cnt_hijo, epsilon_h, True)
                h_coords = [(pt[0][0] * escala, -pt[0][1] * escala) for pt in approx_h]
                
                if len(h_coords) >= 3:
                    agujeros_coords.append(h_coords)
            
            idx_hijo = hierarchy[idx_hijo][0] # Siguiente hermano

        # --- Crear Polígono Shapely ---
        poly = Polygon(shell=shell_coords, holes=agujeros_coords)
        
        if not poly.is_valid:
            poly = poly.buffer(0) # Reparar geometría
            
        poligonos_shapely.append(poly)

    print(f"Generando malla de {len(poligonos_shapely)} partes...")

    if not poligonos_shapely:
        print("No se generaron polígonos válidos.")
        return

    # 4. Extrusión
    mallas = []
    for poly in poligonos_shapely:
        mesh = trimesh.creation.extrude_polygon(poly, height=altura_mm)
        mallas.append(mesh)

    mesh_final = trimesh.util.concatenate(mallas)

    # 5. VISUALIZACIÓN
    print("------------------------------------------------")
    print(">> Abriendo visor 3D interactivo...")
    print(">> Usa el MOUSE para rotar. Cierra la ventana para continuar y guardar.")
    print("------------------------------------------------")
    
    # Esta línea abre la ventana emergente
    mesh_final.show() 

    # 6. Exportar
    mesh_final.export(ruta_salida_stl)
    print(f"¡Guardado! Archivo STL en: {ruta_salida_stl}")

# --- Ejecución ---
archivo_entrada = r'ChatGPT Image 11 dic 2025, 11_43_56.png'
archivo_salida = 'modelo_final_vis.stl'

if os.path.exists(archivo_entrada):
    generar_stl_con_visor(archivo_entrada, archivo_salida, altura_mm=5.0, escala=0.15)
else:
    print(f"No encuentro el archivo: {archivo_entrada}")