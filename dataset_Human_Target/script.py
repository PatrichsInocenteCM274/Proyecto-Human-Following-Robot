from PIL import Image
import os
import random
import math

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = "./"

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_imagenes)

# Filtrar solo las imágenes PNG
imagenes = []
for archivo in archivos:
    if archivo.endswith(".png"):
        ruta_imagen = os.path.join(carpeta_imagenes, archivo)
        if os.path.isfile(ruta_imagen):
            imagen = Image.open(ruta_imagen)
            imagenes.append(imagen)

# Seleccionar 30 imágenes aleatorias
cantidad_imagenes_seleccionadas = 30
imagenes_seleccionadas = random.sample(imagenes, cantidad_imagenes_seleccionadas)

# Calcular el tamaño de la matriz
lado_matriz = math.ceil(math.sqrt(cantidad_imagenes_seleccionadas))
ancho_colage = lado_matriz * max(imagen.width for imagen in imagenes_seleccionadas)
alto_colage = lado_matriz * max(imagen.height for imagen in imagenes_seleccionadas)

# Crear una nueva imagen con las dimensiones del collage
colage = Image.new("RGB", (ancho_colage, alto_colage))

# Pegar cada imagen en la posición correspondiente de la matriz
posicion_x = 0
posicion_y = 0
for imagen in imagenes_seleccionadas:
    colage.paste(imagen, (posicion_x, posicion_y))
    posicion_x += imagen.width
    if posicion_x >= ancho_colage:
        posicion_x = 0
        posicion_y += imagen.height

# Guardar el collage
colage.save("colage.jpg")

