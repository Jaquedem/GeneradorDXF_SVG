import cv2
import numpy as np
import svgwrite
import os

def generar_svg_limpio(ruta_imagen_entrada, ruta_salida_svg):
    print(f"Procesando y limpiando: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None:
        print("Error: No se carga la imagen.")
        return

    # 1. Preprocesamiento (Regresamos a Thresholding para tener formas sólidas)
    # Canny es el culpable de las dobles líneas en este caso, mejor usar binarización inteligente.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Blur un poco más fuerte para unir texturas

    # Binarización Invertida (Objeto blanco, fondo negro)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Encontrar contornos CON jerarquía
    # RETR_TREE es vital aquí para saber qué contorno está dentro de cuál
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contornos brutos detectados: {len(contours)}")

    if hierarchy is None:
        print("No se encontraron contornos.")
        return

    # La jerarquía viene en formato: [Next, Previous, First_Child, Parent]
    hierarchy = hierarchy[0] 

    contornos_validos = []

    # 3. FILTRO LÓGICO: Eliminar duplicados encimados
    for i, cnt in enumerate(contours):
        area_actual = cv2.contourArea(cnt)
        
        # Filtro 1: Eliminar ruido minúsculo
        if area_actual < 50: 
            continue

        padre_idx = hierarchy[i][3] # Índice del padre

        es_valido = True

        if padre_idx != -1:
            # Si tiene padre, comparamos áreas
            area_padre = cv2.contourArea(contours[padre_idx])
            
            # SI el área del hijo es más del 80% del área del padre, es una doble línea (ruido)
            if area_padre > 0:
                ratio = area_actual / area_padre
                if ratio > 0.85: # Ajustable: 0.85 significa "si son casi iguales, bórralo"
                    es_valido = False
        
        if es_valido:
            contornos_validos.append(cnt)

    print(f"Contornos limpios finales: {len(contornos_validos)}")

    # 4. Generar SVG
    height, width = img.shape[:2]
    dwg = svgwrite.Drawing(ruta_salida_svg, profile='full', size=(width, height))
    
    # Usamos stroke fino rojo para ver bien el corte
    main_group = dwg.g(stroke="red", stroke_width=1, fill="none")

    for cnt in contornos_validos:
        # Suavizado inteligente
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        points = [tuple(pt[0]) for pt in approx]
        
        if len(points) > 2:
            path_data = ["M", f"{points[0][0]},{points[0][1]}"]
            for p in points[1:]:
                path_data.append(f"L {p[0]},{p[1]}")
            path_data.append("Z")
            
            main_group.add(dwg.path(d=" ".join(path_data)))

    dwg.add(main_group)
    dwg.save()
    print(f"¡Listo! SVG limpio guardado en: {ruta_salida_svg}")

# --- Ejecución ---
archivo_entrada = r'ChatGPT Image 11 dic 2025, 11_43_56.png'
archivo_salida = 'resultado_limpio.svg'

if os.path.exists(archivo_entrada):
    generar_svg_limpio(archivo_entrada, archivo_salida)
else:
    print(f"Archivo no encontrado: {archivo_entrada}")