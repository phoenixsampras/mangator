import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from colorizator import MangaColorizator
from zoom import upscale_image

def process_image(image, colorizator, args):
    colorizator.set_image(image, args.size, args.denoiser, args.denoiser_sigma)
    result = colorizator.colorize()
    
    # Aplicar zoom si es diferente de 1
    if args.zoom != 1.0:
        if args.ai_upscale:
            # Usar AI upscaling
            result = upscale_image(result, args.zoom, device='cuda' if args.gpu else 'cpu')
        else:
            # Usar resize tradicional
            h, w = result.shape[:2]
            new_h = int(h * args.zoom)
            new_w = int(w * args.zoom)
            result = resize_image(result, (new_w, new_h))
    
    return result
    
def colorize_single_image(image_path, save_path, colorizator, args):
    try:
        image = plt.imread(image_path)
        colorization = process_image(image, colorizator, args)
        
        # Usar PIL en lugar de plt.imsave para mejor control de calidad
        if isinstance(colorization, np.ndarray):
            img = Image.fromarray(colorization)
            img.save(save_path, 'PNG', quality=100, optimize=False)
        else:
            colorization.save(save_path, 'PNG', quality=100, optimize=False)
            
        return True
    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return False
    

def colorize_images(target_path, colorizator, args):
    images = os.listdir(args.path)
    
    for image_name in images:
        file_path = os.path.join(args.path, image_name)
        
        if os.path.isdir(file_path):
            continue
        
        name, ext = os.path.splitext(image_name)
        if (ext != '.png'):
            image_name = name + '.png'
        
        print(file_path)
        
        save_path = os.path.join(target_path, image_name)
        colorize_single_image(file_path, save_path, colorizator, args)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Coloriza imágenes manga en blanco y negro usando inteligencia artificial.
        
        Ejemplos de uso:
            # Colorizar una imagen individual:
            python inference.py -p manga.png -g
            
            # Colorizar con zoom x2:
            python inference.py -p manga.png -z 2
            
            # Colorizar una carpeta completa usando GPU y mayor resolución:
            python inference.py -p ./carpeta_manga -g -s 832
            
            # Colorizar sin denoiser y con zoom x1.5:
            python inference.py -p manga.png -nd -z 1.5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-p", "--path", required=True,
                      help="Ruta a la imagen o carpeta a colorizar")
    
    parser.add_argument("-gen", "--generator", default='networks/generator.zip',
                      help="Ruta al modelo generator.zip (default: networks/generator.zip)")
    
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true',
                      help="Usar GPU para el procesamiento (más rápido si hay GPU disponible)")
    
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false',
                      help="Desactivar el denoiser (por defecto está activado)")
    
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25,
                      help="Intensidad del denoiser, valores más altos = más suavizado (default: 25)")
    
    parser.add_argument("-s", "--size", type=int, default=576,
                      help="Tamaño de procesamiento. Debe ser múltiplo de 32. Mayor = mejor calidad pero más lento (default: 576)")
    
    parser.add_argument("-e", "--extractor", default='denoising/models/',
                      help="Ruta al modelo denoiser (default: denoising/models/)")
    
    parser.add_argument("-z", "--zoom", type=float, default=1.0,
                      help="Factor de zoom para la imagen final (ej: 1.5, 2, 2.5). Default: 1.0")
    
    parser.add_argument("-ai", "--ai_upscale", action='store_true',
                      help="Usar AI para el upscaling (más lento pero mejor calidad)")
    
    parser.set_defaults(gpu=False)
    parser.set_defaults(denoiser=True)
    args = parser.parse_args()
    
    if args.size % 32 != 0:
        parser.error("El parámetro size debe ser múltiplo de 32")
    
    if args.zoom < 0.1 or args.zoom > 4.0:
        parser.error("El zoom debe estar entre 0.1 y 4.0")
    
    return args

    
if __name__ == "__main__":
    
    args = parse_args()
    
    device = 'cuda' if args.gpu else 'cpu'
    
    colorizer = MangaColorizator(device, args.generator, args.extractor)
    
    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        os.makedirs(colorization_path, exist_ok=True)
              
        colorize_images(colorization_path, colorizer, args)
        
    elif os.path.isfile(args.path):
        
        split = os.path.splitext(args.path)
        
        if split[1].lower() in ('.jpg', '.png', '.jpeg'):
            # Agregar información del zoom al nombre si se usó
            zoom_info = f'_x{args.zoom}' if args.zoom != 1.0 else ''
            new_image_path = f"{split[0]}_colorized{zoom_info}.png"
            
            if colorize_single_image(args.path, new_image_path, colorizer, args):
                print(f"Imagen guardada como: {new_image_path}")
        else:
            print('Formato de imagen no soportado')
    else:
        print('Ruta inválida')
    
