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
                 color_intensity: float = 1.0):
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

    def load_pdf(self, pdf_path: str) -> List[fitz.Page]:
        """
        Carga un archivo PDF y retorna sus páginas.
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Lista de páginas del PDF
        """
        try:
            doc = fitz.open(pdf_path)
            return [page for page in doc]
        except Exception as e:
            raise Exception(f"Error al cargar el PDF: {str(e)}")

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

    def process_page(self, page: fitz.Page) -> Image.Image:
        """
        Procesa una página con mejor preservación de texto y bordes.
        """
        pix = page.get_pixmap()
        img_data = pix.samples
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        image = np.array(img)
        original_size = image.shape[:2]  # Guardamos el tamaño original

        # Mejorar la detección de texto
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Usar un kernel más pequeño para mejor detección de bordes
        kernel = np.ones((2,2), np.uint8)
        # Aplicar un umbral adaptativo para mejor detección de texto
        text_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        # Dilatar ligeramente para incluir bordes
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # Mejorar la transición entre texto y color
        if self.preserve_text > 0:
            # Crear un borde suave alrededor del texto
            text_mask_blur = cv2.GaussianBlur(text_mask, (3,3), 0)
            text_mask = cv2.addWeighted(text_mask, 0.7, text_mask_blur, 0.3, 0)

        # Colorizar la imagen base
        self.colorizer.set_image(image, self.size, self.denoiser, self.denoiser_sigma)
        colorized = self.colorizer.colorize()
        
        # Ajustar intensidad de color
        colorized = self.adjust_color_intensity(colorized)
        
        # Redimensionar la imagen colorizada al tamaño original
        colorized = cv2.resize(colorized, (original_size[1], original_size[0]))
        
        # Aplicar zoom si es necesario (ahora después de la redimensión)
        if self.zoom != 1.0:
            if self.ai_upscale:
                colorized = upscale_image(colorized, self.zoom, device=self.device)
                # Redimensionar la imagen original y la máscara al nuevo tamaño
                new_h, new_w = colorized.shape[:2]
                image = cv2.resize(image, (new_w, new_h))
                text_mask = cv2.resize(text_mask, (new_w, new_h))
            else:
                h, w = original_size
                new_h = int(h * self.zoom)
                new_w = int(w * self.zoom)
                colorized = cv2.resize(colorized, (new_w, new_h))
                image = cv2.resize(image, (new_w, new_h))
                text_mask = cv2.resize(text_mask, (new_w, new_h))

        # Combinar imagen colorizada con texto original
        text_mask = (text_mask / 255.0) * self.preserve_text
        text_mask = np.stack([text_mask] * 3, axis=-1)
        result = colorized * (1 - text_mask) + image * text_mask
        
        return Image.fromarray(result.astype(np.uint8))

    def create_pdf(self, images: List[Image.Image], output_path: str):
        """
        Crea un nuevo PDF con las imágenes procesadas.
        
        Args:
            images: Lista de imágenes procesadas
            output_path: Ruta donde guardar el PDF
        """
        try:
            first_image = images[0]
            first_image.save(output_path, "PDF", save_all=True, append_images=images[1:])
        except Exception as e:
            raise Exception(f"Error al crear el PDF: {str(e)}")

    def process_pdf(self, input_path: str, output_path: str = None) -> str:
        """
        Procesa un archivo PDF completo y guarda copias en temp.
        
        Args:
            input_path: Ruta al PDF de entrada
            output_path: Ruta para el PDF de salida (opcional)
            
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
            print(f"{Fore.GREEN}✓ PDF cargado exitosamente - {total_pages} páginas{Style.RESET_ALL}")
            
            # Procesar cada página
            processed_images = []
            
            for i, page in enumerate(pages, 1):
                page_start_time = time.time()
                
                # Calcular progreso y estimación
                progress = (i - 1) / total_pages * 100
                if i > 1:
                    avg_time_per_page = (time.time() - start_time) / (i - 1)
                    remaining_pages = total_pages - (i - 1)
                    eta = timedelta(seconds=int(avg_time_per_page * remaining_pages))
                    eta_str = f"- ETA: {eta}"
                else:
                    eta_str = ""
                
                print(f"\n{Fore.CYAN}🎨 Procesando página {i}/{total_pages} [{progress:.1f}%] {eta_str}{Style.RESET_ALL}")
                
                processed_image = self.process_page(page)
                
                # Guardar página procesada
                temp_image_path = os.path.join(temp_dir, f"page_{i:03d}.png")
                processed_image.save(temp_image_path, "PNG")
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
                use_gpu: bool = False,
                size: int = 576,
                denoiser: bool = True,
                denoiser_sigma: int = 25,
                zoom: float = 1.0,
                ai_upscale: bool = False,
                preserve_text: float = 0.5,
                color_intensity: float = 1.0) -> str:
    """
    Función de conveniencia para colorizar un PDF.
    
    Args:
        pdf_path: Ruta al PDF a colorizar
        output_path: Ruta de salida (opcional)
        use_gpu: Si usar GPU
        size: Tamaño de procesamiento
        denoiser: Si usar denoiser
        denoiser_sigma: Intensidad del denoiser
        zoom: Factor de zoom
        ai_upscale: Si usar IA para upscaling
        preserve_text: Intensidad de preservación del texto (0.0 a 1.0)
                      0.0 = no preservar, 1.0 = preservar completamente
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
    
    return colorizator.process_pdf(pdf_path, output_path)

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
    parser.add_argument("-ci", "--color_intensity", type=float, default=1.0,
                       help="Factor de intensidad de color (0.5 = pastel, 1.0 = normal, 1.5 = vivido)")
    
    args = parser.parse_args()
    
    # Validar el rango del parámetro
    if args.preserve_text < 0.0 or args.preserve_text > 1.0:
        parser.error("El parámetro preserve_text debe estar entre 0.0 y 1.0")
    
    try:
        colorize_pdf(
            args.pdf_path,
            args.output,
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