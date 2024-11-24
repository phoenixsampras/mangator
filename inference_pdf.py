"""
Módulo para colorizar PDFs de manga en blanco y negro usando IA.
Mantiene las páginas ya coloreadas y procesa solo las páginas en B&N.
"""

import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import cv2
from typing import List, Tuple, Union
from colorizator import MangaColorizator
from zoom import upscale_image
import shutil
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import time
import mediapipe as mp
from PIL import ImageFilter, ImageDraw

# Inicializar colorama para Windows
init()

class PDFColorizator:
    """Clase para manejar la colorización de PDFs de manga."""
    
    def __init__(self, generator_path: str = 'networks/generator.zip',
                 denoiser_path: str = 'denoising/models/',
                 use_gpu: bool = False,
                 size: int = 576,
                 denoiser: bool = True,
                 denoiser_sigma: int = 25,
                 zoom: float = 1.0,
                 ai_upscale: bool = False,
                 preserve_text: float = 0.5,
                 color_intensity: float = 1.0,
                 extraction_quality: float = 2.0):
        """
        Inicializa el colorizador de PDFs.
        
        Args:
            generator_path: Ruta al modelo generador
            denoiser_path: Ruta al modelo denoiser
            use_gpu: Si usar GPU para el procesamiento
            size: Tamaño de procesamiento (múltiplo de 32)
            denoiser: Si usar denoiser
            denoiser_sigma: Intensidad del denoiser
            zoom: Factor de zoom para la imagen final
            ai_upscale: Si usar IA para upscaling
            preserve_text: Intensidad de preservación del texto (0.0 a 1.0)
                          0.0 = no preservar, 1.0 = preservar completamente
            color_intensity: Factor de intensidad de color (0.5 = pastel, 1.0 = normal, 1.5 = vivido)
            extraction_quality: Factor de calidad para extracción
        """
        self.device = 'cuda' if use_gpu else 'cpu'
        self.colorizer = MangaColorizator(self.device, generator_path, denoiser_path)
        self.size = size
        self.denoiser = denoiser
        self.denoiser_sigma = denoiser_sigma
        self.zoom = zoom
        self.ai_upscale = ai_upscale
        self.preserve_text = preserve_text
        self.color_intensity = max(0.5, min(2.0, color_intensity))
        self.extraction_quality = extraction_quality
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.yolo_weights = os.path.join('models', 'yolov4-tiny.weights')
        self.yolo_config = os.path.join('models', 'yolov4-tiny.cfg')

    def load_pdf(self, pdf_path: str) -> List[fitz.Page]:
        """
        Carga un archivo PDF y retorna sus páginas con máxima calidad.
        """
        try:
            doc = fitz.open(pdf_path)
            return [page for page in doc]
        except Exception as e:
            raise Exception(f"Error al cargar el PDF: {str(e)}")

    def extract_page_image(self, page: fitz.Page) -> np.ndarray:
        """
        Extrae la imagen de una página y FUERZA el formato uint8.
        """
        try:
            # Obtener pixmap
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
                colorspace=fitz.csRGB
            )
            
            # MÉTODO 1: Usar PIL como intermediario
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img, dtype=np.uint8)  # Forzar uint8 aquí
            
            # Verificación de seguridad
            if img_array.dtype != np.uint8:
                # MÉTODO 2: Conversión directa si el primero falla
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_array.reshape(pix.height, pix.width, 3)
            
            # Verificación final
            if img_array.dtype != np.uint8:
                # MÉTODO 3: Conversión manual si todo lo demás falla
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                else:
                    img_array = img_array.clip(0, 255).astype(np.uint8)
            
            print(f"Debug - Final image: dtype={img_array.dtype}, shape={img_array.shape}, range=[{img_array.min()}, {img_array.max()}]")
            
            # Última verificación
            assert img_array.dtype == np.uint8, f"Failed to convert to uint8, got {img_array.dtype}"
            assert img_array.max() <= 255, f"Values out of range: max={img_array.max()}"
            assert img_array.min() >= 0, f"Values out of range: min={img_array.min()}"
            
            return img_array
            
        except Exception as e:
            print(f"💥 Error en extract_page_image:")
            print(f"- Error type: {type(e)}")
            print(f"- Error message: {str(e)}")
            raise Exception(f"Error al extraer imagen: {str(e)}")

    def is_colored_page(self, image: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Determina si una página está en color o en B&N.
        
        Args:
            image: Imagen en formato numpy array
            threshold: Umbral para determinar si es color
            
        Returns:
            True si la página está en color, False si es B&N
        """
        if len(image.shape) == 3:  # Si tiene 3 canales
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation)
            return mean_saturation > threshold * 255
        return False

    def adjust_color_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Ajusta la intensidad del color de la imagen.
        """
        # Convertir a HSV para manipular la saturación
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        if self.color_intensity != 1.0:
            # Ajustar saturación
            hsv[:, :, 1] = hsv[:, :, 1] * self.color_intensity
            # Ajustar valor para colores más vividos o pasteles
            if self.color_intensity > 1.0:
                hsv[:, :, 2] = hsv[:, :, 2] * (1 + (self.color_intensity - 1) * 0.5)
            else:
                hsv[:, :, 2] = hsv[:, :, 2] * (1 + (1 - self.color_intensity) * 0.3)
            
            # Asegurar que los valores estén en rango
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # Volver a RGB
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def detect_text_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta regiones de texto en manga usando un enfoque específico para bocadillos.
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Pre-procesamiento para resaltar los bocadillos
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # Detectar áreas muy blancas
        
        # 2. Encontrar contornos de áreas blancas (potenciales bocadillos)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Crear máscara inicial
        text_mask = np.zeros_like(gray)
        
        # 4. Filtrar y validar contornos
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Ignorar áreas muy pequeñas
                continue
            
            # Obtener el rectángulo que contiene el contorno
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filtros para validar bocadillos:
            # - Debe ser relativamente rectangular
            aspect_ratio = w / float(h)
            if not (0.5 < aspect_ratio < 4.0):
                continue
            
            # - Debe tener un alto contraste interno (texto negro sobre blanco)
            roi = gray[y:y+h, x:x+w]
            if roi.mean() < 200 or roi.std() < 30:  # Debe ser mayormente blanco con variación
                continue
            
            # - Debe tener bordes definidos
            roi_edges = cv2.Canny(roi, 100, 200)
            if cv2.countNonZero(roi_edges) < (w * h * 0.1):  # Mínimo de bordes
                continue
            
            # Si pasa todos los filtros, es probablemente un bocadillo
            cv2.drawContours(text_mask, [cnt], -1, (255), -1)
        
        # 5. Limpiar la máscara
        kernel = np.ones((3,3), np.uint8)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
        
        return text_mask

    def detect_manga_characters(self, image: np.ndarray, temp_dir: str, page_num: int) -> np.ndarray:
        """
        Detector de personajes manga usando PIL en lugar de OpenCV.
        """
        try:
            # Convertir numpy array a PIL
            pil_image = Image.fromarray(image)
            w, h = pil_image.size
            mask = Image.new('L', (w, h), 0)  # Máscara en blanco
            debug_image = pil_image.copy()
            
            # 1. Convertir a escala de grises y detectar bordes
            gray = pil_image.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            
            # 2. Umbralizar para obtener áreas significativas
            threshold = 128
            binary = edges.point(lambda x: 255 if x > threshold else 0)
            
            # 3. Encontrar áreas de interés
            regions = []
            pixels = binary.load()
            visited = set()
            
            def flood_fill(x, y):
                """Encuentra regiones conectadas."""
                if (x, y) in visited or x < 0 or y < 0 or x >= w or y >= h or pixels[x, y] == 0:
                    return set()
                
                region = {(x, y)}
                visited.add((x, y))
                stack = [(x, y)]
                
                while stack:
                    cx, cy = stack.pop()
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) not in visited and 0 <= nx < w and 0 <= ny < h and pixels[nx, ny] > 0:
                            stack.append((nx, ny))
                            region.add((nx, ny))
                            visited.add((nx, ny))
                
                return region

            # Encontrar regiones significativas
            for y in range(h):
                for x in range(w):
                    if pixels[x, y] > 0 and (x, y) not in visited:
                        region = flood_fill(x, y)
                        if len(region) > (w * h * 0.01):  # Ignorar regiones pequeñas
                            regions.append(region)
            
            # 4. Procesar regiones encontradas
            draw = ImageDraw.Draw(mask)
            debug_draw = ImageDraw.Draw(debug_image)
            
            for region in regions:
                # Obtener bounding box
                xs = [x for x, y in region]
                ys = [y for x, y in region]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                width = x2 - x1
                height = y2 - y1
                
                # Filtrar por proporción típica de personajes manga
                aspect_ratio = height / width if width > 0 else 0
                if 1.0 < aspect_ratio < 3.0:
                    # Expandir área
                    expand_x = int(width * 0.2)
                    expand_y = int(height * 0.1)
                    x1 = max(0, x1 - expand_x)
                    y1 = max(0, y1 - expand_y)
                    x2 = min(w, x2 + expand_x)
                    y2 = min(h, y2 + height + expand_y)
                    
                    # Dibujar en máscara y debug
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                    debug_draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            # 5. Suavizar máscara
            mask = mask.filter(ImageFilter.BoxBlur(20))
            
            # Guardar imágenes de debug
            debug_image.save(os.path.join(temp_dir, f"page_{page_num:03d}_manga_detection.png"))
            mask.save(os.path.join(temp_dir, f"page_{page_num:03d}_manga_mask.png"))
            
            # Convertir máscara a numpy array
            return np.array(mask)
            
        except Exception as e:
            print(f"💥 Error en manga detection: {str(e)}")
            return np.zeros((h, w), dtype=np.uint8)

    def trim_white_borders(self, image: Image.Image) -> Image.Image:
        """
        Recorta bordes blancos usando escala de grises para mayor precisión.
        """
        try:
            # Convertir a escala de grises
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Crear máscara binaria (True donde NO es blanco)
            # Umbral más estricto para blancos (245)
            mask = gray_array < 245
            
            # Encontrar bordes del contenido
            coords = np.argwhere(mask)
            if len(coords) == 0:  # Si la imagen está completamente en blanco
                return image
            
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            
            # Agregar margen mínimo de seguridad (1 pixel)
            margin = 1
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(image.width - 1, x1 + margin)
            y1 = min(image.height - 1, y1 + margin)
            
            # Recortar imagen original (manteniendo el color)
            trimmed = image.crop((x0, y0, x1 + 1, y1 + 1))
            
            # Debug info
            print(f"Recorte: ({x0}, {y0}) -> ({x1}, {y1})")
            print(f"Tamaño original: {image.size} -> Nuevo tamaño: {trimmed.size}")
            
            return trimmed
            
        except Exception as e:
            print(f"💥 Error en trim_white_borders: {str(e)}")
            return image  # Retornar imagen original si algo falla

    def process_page(self, page: fitz.Page, temp_dir: str, page_num: int) -> Image.Image:
        """
        Procesa una página y elimina bordes blancos.
        """
        try:
            # 1. Extraer imagen original
            image = self.extract_page_image(page)
            Image.fromarray(image).save(
                os.path.join(temp_dir, f"page_{page_num:03d}_01_original.png")
            )
            
            # 2. Forzar uint8
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(
                os.path.join(temp_dir, f"page_{page_num:03d}_02_uint8.png")
            )
            
            original_size = image.shape[:2]
            
            # 3. Pre-colorización
            self.colorizer.set_image(image, self.size, self.denoiser, self.denoiser_sigma)
            print(f"Debug - Pre-colorize shape: {image.shape}, dtype: {image.dtype}")
            
            # 4. Colorización
            colorized = self.colorizer.colorize()
            Image.fromarray(colorized.astype(np.uint8)).save(
                os.path.join(temp_dir, f"page_{page_num:03d}_03_colorized_raw.png")
            )
            
            # 5. Resize
            colorized = cv2.resize(colorized, (original_size[1], original_size[0]))
            Image.fromarray(colorized.astype(np.uint8)).save(
                os.path.join(temp_dir, f"page_{page_num:03d}_04_colorized_resized.png")
            )
            
            # 6. Ajuste de color
            colorized = self.adjust_color_intensity(colorized).astype(np.uint8)
            Image.fromarray(colorized).save(
                os.path.join(temp_dir, f"page_{page_num:03d}_05_color_adjusted.png")
            )
            
            # 7. Texto
            if self.preserve_text > 0:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(
                    os.path.join(temp_dir, f"page_{page_num:03d}_06_gray.png"),
                    gray
                )
                
                text_mask = cv2.adaptiveThreshold(
                    gray, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=15,
                    C=2
                )
                cv2.imwrite(
                    os.path.join(temp_dir, f"page_{page_num:03d}_07_text_mask_raw.png"),
                    text_mask
                )
                
                # Dilatar y suavizar
                kernel = np.ones((2,2), np.uint8)
                text_mask = cv2.dilate(text_mask, kernel, iterations=1)
                text_mask = cv2.GaussianBlur(text_mask, (3,3), 0)
                cv2.imwrite(
                    os.path.join(temp_dir, f"page_{page_num:03d}_08_text_mask_processed.png"),
                    text_mask
                )
                
                # Mezclar
                text_mask = text_mask.astype(np.float32) / 255.0
                text_mask = np.power(text_mask, 1.5)
                text_mask = np.stack([text_mask] * 3, axis=-1)
                
                result = (colorized.astype(np.float32) * (1 - text_mask) + 
                         image.astype(np.float32) * text_mask * 0.6)  # Texto aún más negro
                result = result.clip(0, 255).astype(np.uint8)
            else:
                result = colorized
            
            Image.fromarray(result).save(
                os.path.join(temp_dir, f"page_{page_num:03d}_09_final.png")
            )
            
            # Antes de retornar, recortar bordes blancos
            result_image = Image.fromarray(result)
            trimmed_image = self.trim_white_borders(result_image)
            
            # Guardar versión final recortada
            trimmed_image.save(
                os.path.join(temp_dir, f"page_{page_num:03d}_10_trimmed.png")
            )
            
            print(f"✅ Imágenes intermedias guardadas en {temp_dir}")
            return trimmed_image
            
        except Exception as e:
            print(f"💥 Error procesando página {page_num}:")
            print(f"Error details: {str(e)}")
            raise

    def create_pdf(self, images: List[Image.Image], output_path: str):
        """
        Crea un nuevo PDF manteniendo el tamaño de página completo.
        """
        try:
            doc = fitz.open()
            
            for img in images:
                # Calcular tamaño de página en puntos (72 puntos = 1 pulgada)
                # Para papel Letter (8.5 x 11 pulgadas)
                page_width = 8.5 * 72  # 612 puntos
                page_height = 11 * 72  # 792 puntos
                
                # Crear página con tamaño Letter
                page = doc.new_page(width=page_width, height=page_height)
                
                # Calcular dimensiones para ajustar la imagen al ancho de la página
                # dejando márgenes de 0.5 pulgadas en cada lado
                margin = 0  # 36 puntos de margen
                available_width = page_width - (2 * margin)
                available_height = page_height - (2 * margin)
                
                # Calcular factor de escala manteniendo proporción
                scale = min(available_width / img.width, available_height / img.height)
                new_width = img.width * scale
                new_height = img.height * scale
                
                # Centrar la imagen en la página
                x0 = (page_width - new_width) / 2
                y0 = (page_height - new_height) / 2
                rect = fitz.Rect(x0, y0, x0 + new_width, y0 + new_height)
                
                # Convertir imagen a bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG', optimize=False)
                img_bytes = img_bytes.getvalue()
                
                # Insertar imagen centrada y escalada
                page.insert_image(rect, stream=img_bytes)

            # Guardar PDF con configuración optimizada
            doc.save(
                output_path,
                garbage=4,
                deflate=False,
                clean=True,
                pretty=True
            )
            doc.close()
            
            print(f"{Fore.GREEN}✨ PDF creado con tamaño de página completo: {output_path}{Style.RESET_ALL}")
            
        except Exception as e:
            raise Exception(f"Error al crear el PDF: {str(e)}")

    def process_pdf(self, input_path: str, output_path: str = None, start_page: int = None, end_page: int = None) -> str:
        """
        Procesa un archivo PDF completo y guarda copias en temp.
        
        Args:
            input_path: Ruta al PDF de entrada
            output_path: Ruta para el PDF de salida (opcional)
            start_page: Número de página inicial (1-based, opcional)
            end_page: Número de página final (1-based, opcional)
            
        Returns:
            Ruta al PDF procesado
        """
        print(f"\n{Fore.CYAN}🚀 Iniciando proceso de colorización...{Style.RESET_ALL}")
        
        start_time = time.time()
        if not output_path:
            base_path = os.path.splitext(input_path)[0]
            output_path = f"{base_path}_colorized.pdf"

        # Crear directorio temp si no existe
        temp_dir = os.path.join(os.path.dirname(input_path), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        print(f"{Fore.YELLOW}📁 Directorio temporal creado en: {temp_dir}{Style.RESET_ALL}")

        try:
            # Cargar páginas
            print(f"{Fore.CYAN}📖 Cargando PDF...{Style.RESET_ALL}")
            pages = self.load_pdf(input_path)
            total_pages = len(pages)

            # Validar y ajustar el rango de páginas
            start_page = max(1, start_page if start_page else 1)
            end_page = min(total_pages, end_page if end_page else total_pages)
            
            if start_page > end_page:
                raise ValueError(f"Página inicial ({start_page}) no puede ser mayor que página final ({end_page})")
            
            print(f"{Fore.GREEN}✓ PDF cargado exitosamente - Procesando páginas {start_page} a {end_page} de {total_pages}{Style.RESET_ALL}")
            
            # Procesar cada página
            processed_images = []
            
            # Mantener todas las páginas, pero procesar solo el rango seleccionado
            for i in range(1, total_pages + 1):
                if i < start_page or i > end_page:
                    # Copiar página original sin procesar
                    pix = pages[i-1].get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    processed_images.append(img)
                    print(f"{Fore.YELLOW}⏩ Saltando página {i} (fuera del rango){Style.RESET_ALL}")
                    continue

                page_start_time = time.time()
                
                # Calcular progreso y estimación
                progress = (i - start_page) / (end_page - start_page + 1) * 100
                if i > start_page:
                    avg_time_per_page = (time.time() - start_time) / (i - start_page)
                    remaining_pages = end_page - i + 1
                    eta = timedelta(seconds=int(avg_time_per_page * remaining_pages))
                    eta_str = f"- ETA: {eta}"
                else:
                    eta_str = ""
                
                print(f"\n{Fore.CYAN}🎨 Procesando página {i}/{total_pages} [{progress:.1f}%] {eta_str}{Style.RESET_ALL}")
                
                processed_image = self.process_page(pages[i-1], temp_dir, i)
                processed_images.append(processed_image)
                
                # Tiempo por página
                page_time = time.time() - page_start_time
                print(f"{Fore.GREEN}✓ Página {i} completada en {page_time:.1f} segundos{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}📑 Generando PDF final...{Style.RESET_ALL}")
            # Crear nuevo PDF
            self.create_pdf(processed_images, output_path)
            
            # Guardar copia del PDF en temp con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_pdf_path = os.path.join(temp_dir, f"colorized_{timestamp}.pdf")
            shutil.copy2(output_path, temp_pdf_path)
            
            # Tiempo total
            total_time = time.time() - start_time
            
            # Resumen final
            print(f"\n{Fore.GREEN}✨ ¡Proceso completado exitosamente! ✨{Style.RESET_ALL}")
            print(f"{Fore.GREEN}⏱️  Tiempo total: {timedelta(seconds=int(total_time))}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}📄 PDF guardado como: {output_path}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}💾 Copia de respaldo en: {temp_pdf_path}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}🖼️  Imágenes individuales guardadas en: {temp_dir}{Style.RESET_ALL}")
            
            return output_path
            
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error durante el proceso:{Style.RESET_ALL}")
            print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
            raise Exception(f"Error al procesar el PDF: {str(e)}")

def colorize_pdf(pdf_path: str, 
                output_path: str = None, 
                start_page: int = None,
                end_page: int = None,
                use_gpu: bool = False,
                size: int = 576,
                denoiser: bool = True,
                denoiser_sigma: int = 25,
                zoom: float = 1.0,
                ai_upscale: bool = False,
                preserve_text: float = 0.5,
                color_intensity: float = 1.5) -> str:
    """
    Función de conveniencia para colorizar un PDF.
    
    Args:
        pdf_path: Ruta al PDF a colorizar
        output_path: Ruta de salida (opcional)
        start_page: Número de página inicial (1-based, opcional)
        end_page: Número de página final (1-based, opcional)
        use_gpu: Si usar GPU
        size: Tamaño de procesamiento
        denoiser: Si usar denoiser
        denoiser_sigma: Intensidad del denoiser
        zoom: Factor de zoom
        ai_upscale: Si usar IA para upscaling
        preserve_text: Intensidad de preservación del texto (0.0 a 1.0)
        color_intensity: Factor de intensidad de color (0.5 = pastel, 1.0 = normal, 1.5 = vivido)
    """
    colorizator = PDFColorizator(
        use_gpu=use_gpu,
        size=size,
        denoiser=denoiser,
        denoiser_sigma=denoiser_sigma,
        zoom=zoom,
        ai_upscale=ai_upscale,
        preserve_text=preserve_text,
        color_intensity=color_intensity
    )
    
    return colorizator.process_pdf(pdf_path, output_path, start_page, end_page)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coloriza PDFs de manga en B&N")
    parser.add_argument("pdf_path", help="Ruta al PDF a colorizar")
    parser.add_argument("-o", "--output", help="Ruta de salida para el PDF")
    parser.add_argument("-g", "--gpu", action="store_true", help="Usar GPU")
    parser.add_argument("-s", "--size", type=int, default=576, help="Tamaño de procesamiento")
    parser.add_argument("-nd", "--no_denoise", action="store_false", dest="denoiser", 
                       help="Desactivar denoiser")
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25, 
                       help="Intensidad del denoiser")
    parser.add_argument("-z", "--zoom", type=float, default=1.0, help="Factor de zoom")
    parser.add_argument("-ai", "--ai_upscale", action="store_true", 
                       help="Usar IA para upscaling")
    parser.add_argument("-pt", "--preserve_text", type=float, default=0.5,
                       help="Intensidad de preservación del texto (0.0 a 1.0, default: 0.5)")
    parser.add_argument("-ci", "--color_intensity", type=float, default=1.5,
                       help="Factor de intensidad de color (0.5 = pastel, 1.0 = normal, 1.5 = vivido)")
    parser.add_argument("-sp", "--start_page", type=int,
                       help="Página inicial a procesar (1-based)")
    parser.add_argument("-ep", "--end_page", type=int,
                       help="Página final a procesar (1-based)")
    
    args = parser.parse_args()
    
    # Validar el rango del parámetro
    if args.preserve_text < 0.0 or args.preserve_text > 1.0:
        parser.error("El parámetro preserve_text debe estar entre 0.0 y 1.0")
    
    try:
        colorize_pdf(
            args.pdf_path,
            args.output,
            args.start_page,
            args.end_page,
            args.gpu,
            args.size,
            args.denoiser,
            args.denoiser_sigma,
            args.zoom,
            args.ai_upscale,
            args.preserve_text,
            args.color_intensity
        )
    except Exception as e:
        print(f"Error: {str(e)}")