import cv2
import pytesseract
import os
import csv
import time
import Levenshtein

# === CONFIGURACIÓN ===
DATASET_FOLDER = "dataset_placas"
GT_FILE = os.path.join(DATASET_FOLDER, "ground_truth.txt")
REPORT_FILE = "reporte_final.txt"

# NOTA: En Ubuntu no definimos 'tesseract_cmd' porque ya está en el PATH.

def preprocesar(imagen_path):
    """
    Pipeline Clásico:
    1. Cargar Imagen
    2. Escala de Grises
    3. Desenfoque Gaussiano (eliminar ruido de sal/pimienta)
    4. Binarización (Otsu)
    """
    img = cv2.imread(imagen_path)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold Otsu para separar texto del fondo automáticamente
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def calcular_y_guardar_reporte(resultados, tiempo_total):
    total_imgs = len(resultados)
    if total_imgs == 0:
        print("Error: No se procesaron imágenes.")
        return

    # --- VARIABLES ACUMULADORAS ---
    tp_deteccion = 0  # Placas donde se detectó algún texto
    fn_deteccion = 0  # Placas vacías (fallo de detección)
    
    placas_exactas = 0      # Para PAR (Plate Accuracy)
    total_distancia = 0     # Para Levenshtein
    total_caracteres = 0    # Total letras reales
    total_errores = 0       # S + D + I (Para CER)
    
    for res in resultados:
        real = res['real']
        pred = res['pred']
        dist = res['levenshtein']
        
        # 1. Detección (Simulada para recorte)
        # Si Tesseract devolvió al menos 1 caracter, contamos como detección positiva
        if len(pred) > 0:
            tp_deteccion += 1
        else:
            fn_deteccion += 1
            
        # 2. OCR (Reconocimiento)
        if real == pred:
            placas_exactas += 1
            
        total_distancia += dist
        total_caracteres += len(real)
        total_errores += dist # Levenshtein es la suma de errores de edición

    # --- CÁLCULO DE MÉTRICAS ---
    
    # METRICAS DETECCIÓN
    # Precision: En dataset recortado, asumimos que no hay falsos positivos de fondo.
    precision = 1.0 
    recall = tp_deteccion / total_imgs
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # METRICAS OCR
    par = placas_exactas / total_imgs  # Plate Accuracy Rate
    cer = total_errores / total_caracteres if total_caracteres > 0 else 0
    car = 1.0 - cer  # Character Accuracy Rate (aprox)
    wer = (total_imgs - placas_exactas) / total_imgs # Word Error Rate (1 placa = 1 palabra)
    avg_levenshtein = total_distancia / total_imgs
    e2e_accuracy = par # En este pipeline secuencial, el E2E es igual al PAR final

    # RENDIMIENTO
    fps = total_imgs / tiempo_total
    ms_por_img = (tiempo_total / total_imgs) * 1000

    # --- GENERACIÓN DEL TEXTO DEL REPORTE ---
    lines = []
    lines.append("="*50)
    lines.append(f"REPORTE DE EVALUACIÓN: OPENCV + TESSERACT (UBUNTU)")
    lines.append("="*50)
    lines.append(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Imágenes: {total_imgs}")
    lines.append("-" * 50)
    
    lines.append("1. MÉTRICAS DE DETECCIÓN")
    lines.append(f"   Precision (P):    {precision:.4f} (Asumido 1.0 en imgs recortadas)")
    lines.append(f"   Recall (R):       {recall:.4f} (Capacidad de leer algo)")
    lines.append(f"   F1-Score:         {f1_score:.4f}")
    lines.append(f"   IoU / mAP:        N/A (Dataset ya recortado - IoU=1.0 implícito)")
    lines.append("-" * 50)
    
    lines.append("2. MÉTRICAS DE OCR (RECONOCIMIENTO)")
    lines.append(f"   PAR (Plate Accuracy):     {par*100:.2f}%  <-- Placas perfectas")
    lines.append(f"   CAR (Character Accuracy): {car*100:.2f}%")
    lines.append(f"   CER (Character Error):    {cer*100:.2f}%  <-- Error por letra")
    lines.append(f"   WER (Word Error Rate):    {wer*100:.2f}%")
    lines.append(f"   Levenshtein Promedio:     {avg_levenshtein:.2f} ediciones/placa")
    lines.append("-" * 50)
    
    lines.append("3. PIPELINE & RENDIMIENTO")
    lines.append(f"   E2E Accuracy:     {e2e_accuracy*100:.2f}%")
    lines.append(f"   Tiempo Total:     {tiempo_total:.2f} seg")
    lines.append(f"   FPS (Velocidad):  {fps:.2f} frames/seg")
    lines.append(f"   Latencia:         {ms_por_img:.2f} ms/imagen")
    lines.append("="*50)

    reporte_texto = "\n".join(lines)
    
    # Guardar a archivo
    with open(REPORT_FILE, 'w') as f:
        f.write(reporte_texto)
        
    print("\n" + reporte_texto)
    print(f"\n[INFO] Reporte detallado guardado en '{REPORT_FILE}'")

def main():
    if not os.path.exists(GT_FILE):
        print(f"Error: No se encuentra '{GT_FILE}'. Ejecuta primero el generador.")
        return

    print(f"--- Iniciando Evaluación (OpenCV + Tesseract) ---")
    print("Cargando imágenes y procesando... (Esto puede tardar unos segundos)")

    # Configuración Tesseract: PSM 7 (Línea simple), Whitelist (Alfanumérico)
    custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    
    resultados = []
    
    # Leer lista de imágenes
    datos_dataset = []
    with open(GT_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Saltar cabecera
        for row in reader:
            datos_dataset.append(row)

    start_time = time.time()

    for i, (nombre_archivo, texto_real) in enumerate(datos_dataset):
        ruta = os.path.join(DATASET_FOLDER, nombre_archivo)
        
        # 1. Preprocesar
        img_proc = preprocesar(ruta)
        
        if img_proc is not None:
            # 2. Inferencia OCR
            try:
                texto_pred = pytesseract.image_to_string(img_proc, config=custom_config)
                # Limpieza de string (Upper, quitar espacios y saltos)
                texto_pred = texto_pred.strip().upper().replace(" ", "").replace("\n", "")
            except Exception as e:
                print(f"Error Tesseract en {nombre_archivo}: {e}")
                texto_pred = ""
            
            # 3. Calcular distancia individual
            dist = Levenshtein.distance(texto_real, texto_pred)
            
            resultados.append({
                'real': texto_real,
                'pred': texto_pred,
                'levenshtein': dist
            })
        
        # Barra de progreso simple
        if i % 50 == 0:
            print(f"Procesando: {i}/{len(datos_dataset)}", end='\r')

    end_time = time.time()
    
    calcular_y_guardar_reporte(resultados, end_time - start_time)

if __name__ == "__main__":
    main()