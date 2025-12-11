import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import os

# Variable global para almacenar el estado de selección
seleccionados = []

def on_pick(event):
    """Manejador de eventos de clic en los polígonos"""
    artista = event.artist
    indice = artista.get_label() # Usamos la etiqueta para guardar el índice
    indice = int(indice)
    
    # Toggle (Alternar estado)
    if seleccionados[indice]['activo']:
        seleccionados[indice]['activo'] = False
        artista.set_facecolor('gray')
        artista.set_alpha(0.3)
    else:
        seleccionados[indice]['activo'] = True
        artista.set_facecolor('#00ff00') # Verde brillante
        artista.set_alpha(0.7)
    
    event.canvas.draw()

def editor_y_extrusion(ruta_imagen_entrada, ruta_salida_stl, altura_mm=4.0, escala=0.15):
    global seleccionados
    print(f"Cargando editor para: {ruta_imagen_entrada}...")
    
    img = cv2.imread(ruta_imagen_entrada)
    if img is None: return

    # 1. Procesamiento (Binarización inteligente)
    # Usamos canal verde para mejor contraste en objetos de color
    canal = img[:, :, 1] if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(canal, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Limpieza morfológica básica para unir líneas rotas antes de editar
    kernel = np.ones((3,3), np.uint8)
    mascara = cv2.dilate(thresh, kernel, iterations=1)
    mascara = cv2.erode(mascara, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return
    hierarchy = hierarchy[0]

    # 2. Preparar datos para el Editor
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("EDITOR DE VECTORES\nClic: Activar/Desactivar | Cerrar ventana: Generar STL")
    ax.set_aspect('equal')
    # Invertir eje Y para visualización correcta
    ax.invert_yaxis() 

    seleccionados = [] # Reiniciar lista global

    print(f">> Se detectaron {len(contours)} contornos.")
    print(">> Selecciona en la ventana lo que quieras conservar.")

    for i, cnt in enumerate(contours):
        # Filtro inicial de basura muy pequeña
        if cv2.contourArea(cnt) < 50:
            seleccionados.append({'poly': None, 'activo': False, 'jerarquia': hierarchy[i]})
            continue

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Puntos para Matplotlib (x, y)
        puntos_mpl = [tuple(pt[0]) for pt in approx]
        
        # Lógica inicial: Si tiene padre, es hueco (probablemente queremos mantenerlo si el padre está activo)
        # Por defecto activamos todo lo que sea grande, el usuario decidirá
        estado_inicial = True
        color_inicial = '#00ff00' if estado_inicial else 'gray'
        alpha_inicial = 0.7 if estado_inicial else 0.3

        # Crear Polígono visual
        poly_patch = MplPolygon(puntos_mpl, closed=True, 
                                facecolor=color_inicial, edgecolor='black', 
                                alpha=alpha_inicial, picker=True, label=str(i))
        
        ax.add_patch(poly_patch)
        
        # Guardar datos para procesamiento posterior
        # Guardamos puntos escalados e invertidos para el STL final
        puntos_shapely = [(pt[0][0] * escala, -pt[0][1] * escala) for pt in approx]
        seleccionados.append({
            'poly_pts': puntos_shapely, 
            'activo': estado_inicial, 
            'jerarquia': hierarchy[i]
        })

    # Ajustar límites del gráfico
    h, w = img.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # Conectar evento de clic
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # MOSTRAR EDITOR (El código se pausa aquí hasta cerrar la ventana)
    plt.show()

    print(">> Editor cerrado. Procesando selección...")

    # 3. Generación de Geometría basada en selección
    poligonos_shapely = []
    
    # Recorremos la lista que el usuario modificó
    for i, item in enumerate(seleccionados):
        if not item.get('poly_pts') or not item['activo']:
            continue
            
        # Jerarquia: [Next, Previous, First_Child, Parent]
        padre_idx = item['jerarquia'][3]
        
        # Si tiene padre y el padre TAMBIÉN está activo, entonces este es un AGUJERO
        # Si no tiene padre activo, lo tratamos como un sólido independiente
        es_agujero = False
        if padre_idx != -1:
             # Verificamos si el padre está en la lista de activos
             if seleccionados[padre_idx]['activo']:
                 es_agujero = True
        
        # Para simplificar la extrusión con shapely/trimesh directo:
        # Shapely necesita (Shell, Holes). Es difícil reconstruir eso dinámicamente si el usuario
        # borra padres arbitrariamente.
        # ESTRATEGIA ROBUSTA:
        # Extruimos todo como sólido. Luego usamos operaciones booleanas o simplemente
        # confiamos en que el usuario seleccionó bien.
        
        # MEJOR ESTRATEGIA:
        # Si es un contorno externo (o decidimos tratarlo como sólido), creamos el poligono
        # Y buscamos sus hijos ACTIVOS para hacerlos agujeros.
        
        if not es_agujero: # Es un cuerpo principal
            shell = item['poly_pts']
            holes = []
            
            # Buscar hijos directos en la estructura original que estén ACTIVOS
            # (Iteración simple, podría optimizarse)
            child_idx = item['jerarquia'][2]
            while child_idx != -1:
                child_item = seleccionados[child_idx]
                if child_item['activo'] and child_item.get('poly_pts'):
                    holes.append(child_item['poly_pts'])
                child_idx = seleccionados[child_idx]['jerarquia'][0] # Siguiente hermano

            # Crear objeto Shapely
            try:
                poly = Polygon(shell=shell, holes=holes)
                if not poly.is_valid: poly = poly.buffer(0)
                poligonos_shapely.append(poly)
            except Exception as e:
                print(f"Error en polígono {i}: {e}")

    if not poligonos_shapely:
        print("No seleccionaste ningún polígono válido.")
        return

    # 4. Extrusión
    mallas = []
    for poly in poligonos_shapely:
        mesh = trimesh.creation.extrude_polygon(poly, height=altura_mm)
        mallas.append(mesh)

    if mallas:
        mesh_final = trimesh.util.concatenate(mallas)
        mesh_final.export(ruta_salida_stl)
        print(f"¡STL Generado!: {ruta_salida_stl}")
        mesh_final.show() # Visualizar resultado final 3D
    else:
        print("Error al generar mallas.")

# --- Ejecución ---
archivo_entrada = r'ChatGPT Image 11 dic 2025, 11_43_56.png'
archivo_salida = 'modelo_seleccionado.stl'

if os.path.exists(archivo_entrada):
    editor_y_extrusion(archivo_entrada, archivo_salida, altura_mm=4.0, escala=0.15)
else:
    print("Archivo no encontrado")