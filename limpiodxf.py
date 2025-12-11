import cv2
import numpy as np
import ezdxf  # Librería para DXF
import os

def generar_dxf_limpio(ruta_imagen_entrada, ruta_salida_dxf):
    print(f"Procesando y limpiando para DXF: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None:
        print("Error: No se carga la imagen.")
        return

    # 1. Preprocesamiento (Misma lógica que tu script anterior)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Binarización Invertida
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Encontrar contornos CON jerarquía
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contornos brutos detectados: {len(contours)}")

    if hierarchy is None:
        print("No se encontraron contornos.")
        return

    hierarchy = hierarchy[0] 
    contornos_validos = []

    # 3. FILTRO LÓGICO: Eliminar duplicados encimados
    # (Mantenemos tu lógica intacta aquí)
    for i, cnt in enumerate(contours):
        area_actual = cv2.contourArea(cnt)
        
        if area_actual < 50: 
            continue

        padre_idx = hierarchy[i][3]
        es_valido = True

        if padre_idx != -1:
            area_padre = cv2.contourArea(contours[padre_idx])
            
            if area_padre > 0:
                ratio = area_actual / area_padre
                # Si el hijo es casi igual al padre (>85%), es basura/doble línea
                if ratio > 0.85: 
                    es_valido = False
        
        if es_valido:
            contornos_validos.append(cnt)

    print(f"Contornos limpios finales: {len(contornos_validos)}")

    # 4. Generar DXF (Aquí está el cambio principal)
    # Creamos un documento DXF versión 2010 (muy compatible)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for cnt in contornos_validos:
        # Suavizado inteligente
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Preparamos los puntos para DXF
        # IMPORTANTE: Invertimos el eje Y (-pt[0][1]) porque en imágenes el (0,0) 
        # está arriba-izquierda, y en CAD está abajo-izquierda.
        puntos_dxf = [(float(pt[0][0]), -float(pt[0][1])) for pt in approx]
        
        if len(puntos_dxf) > 2:
            # LWPOLYLINE es la entidad óptima para cortes continuos
            # close=True cierra la figura automáticamente
            msp.add_lwpolyline(puntos_dxf, close=True, dxfattribs={'layer': 'CORTE', 'color': 1})

    doc.saveas(ruta_salida_dxf)
    print(f"¡Listo! DXF limpio guardado en: {ruta_salida_dxf}")

# --- Ejecución ---
archivo_entrada = r'ChatGPT Image 11 dic 2025, 11_29_31.png'
archivo_salida = 'resultado_limpio2.dxf'

if os.path.exists(archivo_entrada):
    generar_dxf_limpio(archivo_entrada, archivo_salida)
else:
    print(f"Archivo no encontrado: {archivo_entrada}")