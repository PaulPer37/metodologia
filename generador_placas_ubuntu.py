import os
import random
import string
import csv
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# === CONFIGURACIÓN ===
OUTPUT_FOLDER = "dataset_placas"
NUM_IMAGENES = 500
ANCHO = 300
ALTO = 100

# Colores (Estilo Ecuador)
COLOR_FONDO = (255, 255, 255)  # Blanco
COLOR_FRANJA = (0, 56, 147)    # Azul
COLOR_TEXTO = (0, 0, 0)        # Negro
COLOR_HEADER = (255, 255, 255) # Blanco

def buscar_fuente_linux():
    """Busca fuentes comunes en Ubuntu para que la placa se vea bien."""
    rutas_posibles = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "arial.ttf" # Por si el usuario instaló msttcorefonts
    ]
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            return ruta
    return None # Usará la por defecto de PIL

def configurar_entorno():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    print(f"--- Iniciando Generador de Placas (Ubuntu) ---")
    print(f"Destino: {OUTPUT_FOLDER}")
    print(f"Cantidad: {NUM_IMAGENES}")

def generar_texto_placa():
    # Formato Ecuatoriano: AAA-0000
    letras = ''.join(random.choices(string.ascii_uppercase, k=3))
    numeros = ''.join(random.choices(string.digits, k=4))
    return f"{letras}-{numeros}"

def aplicar_efectos_realistas(img):
    """Aplica distorsiones para simular una captura real"""
    # 1. Rotación leve (-4 a 4 grados)
    angulo = random.uniform(-4, 4)
    img = img.rotate(angulo, expand=False, fillcolor=COLOR_FONDO)
    
    # 2. Desenfoque (Blur) - Simula cámara en movimiento
    if random.random() > 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    
    # 3. Ruido (Puntos grises/negros)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(50, 400)):
        x = random.randint(0, ANCHO-1)
        y = random.randint(0, ALTO-1)
        color_ruido = random.randint(50, 200)
        draw.point((x, y), fill=(color_ruido, color_ruido, color_ruido))
        
    return img

def crear_placa(idx, font_path):
    texto_placa = generar_texto_placa()
    img = Image.new('RGB', (ANCHO, ALTO), color=COLOR_FONDO)
    draw = ImageDraw.Draw(img)

    # Cargar fuentes
    if font_path:
        font_header = ImageFont.truetype(font_path, 14)
        font_main = ImageFont.truetype(font_path, 60)
    else:
        font_header = ImageFont.load_default()
        font_main = ImageFont.load_default()

    # 1. Dibujar Franja Superior
    alto_franja = 25
    draw.rectangle([(0, 0), (ANCHO, alto_franja)], fill=COLOR_FRANJA)
    
    # 2. Texto "ECUADOR" centrado
    bbox = draw.textbbox((0, 0), "ECUADOR", font=font_header)
    w_h = bbox[2] - bbox[0]
    draw.text(((ANCHO - w_h)/2, 5), "ECUADOR", font=font_header, fill=COLOR_HEADER)

    # 3. Texto de la Placa (AAA-0000)
    bbox_m = draw.textbbox((0, 0), texto_placa, font=font_main)
    w_m = bbox_m[2] - bbox_m[0]
    h_m = bbox_m[3] - bbox_m[1]
    
    x = (ANCHO - w_m) / 2
    y = (ALTO - h_m) / 2 + 10
    draw.text((x, y), texto_placa, font=font_main, fill=COLOR_TEXTO)

    # 4. Aplicar efectos
    img = aplicar_efectos_realistas(img)

    # 5. Guardar archivo
    nombre_archivo = f"placa_{idx:03d}.jpg"
    ruta_completa = os.path.join(OUTPUT_FOLDER, nombre_archivo)
    img.save(ruta_completa)

    return nombre_archivo, texto_placa

def main():
    configurar_entorno()
    fuente_linux = buscar_fuente_linux()
    if fuente_linux:
        print(f"Usando fuente del sistema: {fuente_linux}")
    else:
        print("AVISO: No se detectaron fuentes TTF comunes. Usando fuente por defecto (pixelada).")

    archivo_gt = os.path.join(OUTPUT_FOLDER, "ground_truth.txt")
    
    with open(archivo_gt, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["archivo", "texto_real"]) # Cabecera
        
        for i in range(1, NUM_IMAGENES + 1):
            archivo, texto = crear_placa(i, fuente_linux)
            writer.writerow([archivo, texto])
            
            if i % 50 == 0:
                print(f"Generadas: {i}/{NUM_IMAGENES}...", end='\r')
            
    print(f"\n¡Éxito! Dataset generado en '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()