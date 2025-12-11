import cv2
import numpy as np
import svgwrite
import os

def contornos_alta_precision(ruta_imagen_entrada, ruta_salida_svg):
    print(f"Procesando con alta precisión: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None:
        print("Error: No se encuentra la imagen.")
        return

    # 1. Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Detección de bordes con Canny (El estándar para detalles finos)
    # Ajusta estos dos números si ves mucho ruido o faltan líneas.
    # 50, 150 es un rango estándar. Para imágenes claras, 100, 200 funciona bien.
    edges = cv2.Canny(gray, 100, 200)

    # 3. Encontrar contornos sobre los bordes detectados
    # RETR_CCOMP es bueno para obtener contornos internos y externos
    # CHAIN_APPROX_NONE guarda TODOS los puntos (sin comprimir), máxima fidelidad.
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    print(f"Detalles detectados: {len(contours)}")

    # 4. Configurar SVG
    height, width = img.shape[:2]
    dwg = svgwrite.Drawing(ruta_salida_svg, profile='full', size=(width, height))

    # GRUPO IMPORTANTE:
    # stroke='black': Dibuja la línea negra.
    # stroke_width=1: Línea muy fina para precisión.
    # fill='none': NO rellena, para que no se haga una mancha.
    main_group = dwg.g(stroke="black", stroke_width=1, fill="none")

    for i, cnt in enumerate(contours):
        # Filtro de ruido: eliminamos cosas microscópicas (menores a 10px de área)
        if cv2.contourArea(cnt) < 10 and cv2.arcLength(cnt, True) < 15:
            continue

        # 5. Suavizado MÍNIMO (Para que no se vea pixelado pero respete la forma)
        # Un epsilon muy bajo (0.0005) mantiene la fidelidad casi al 100%
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
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
    print(f"¡Listo! SVG de alta precisión guardado en: {ruta_salida_svg}")

# --- Ejecución ---
archivo_entrada = r'C:\Users\jaqueline.tinoco\Documents\generador de contornos\ChatGPT Image 11 dic 2025, 11_29_31.png'
archivo_salida = 'resultado_fino_canny2.svg'

if os.path.exists(archivo_entrada):
    contornos_alta_precision(archivo_entrada, archivo_salida)
else:
    print(f"No encuentro el archivo: {archivo_entrada}")