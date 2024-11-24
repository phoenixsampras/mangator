import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator
from PIL import Image
import torch  # Import PyTorch to check for GPU availability
import sys
from zoom import upscale_image  # Add this import

def process_image(image, colorizator, size, denoiser, denoiser_sigma, zoom=1.0, gpu=False, ai_upscale=False):
    original_image = image.copy()
    colorizator.set_image(image, size, denoiser, denoiser_sigma)
    colorization = colorizator.colorize(image_to_get_ratio=original_image)
    
    # Add zoom functionality
    if zoom != 1.0:
        if ai_upscale:
            colorization = upscale_image(colorization, zoom, device='cuda' if gpu else 'cpu')
        else:
            h, w = colorization.shape[:2]
            new_h = int(h * zoom)
            new_w = int(w * zoom)
            colorization = Image.fromarray(colorization).resize((new_w, new_h))
            colorization = np.array(colorization)
    
    return colorization

def colorize_single_image(image_path, save_path, colorizator, size, denoiser, denoiser_sigma, zoom=1.0, gpu=False, ai_upscale=False):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        image = plt.imread(image_path)
        colorization = process_image(image, colorizator, size, denoiser, denoiser_sigma, zoom, gpu, ai_upscale)
        colorization = (colorization * 255).astype(np.uint8)
        Image.fromarray(colorization).convert('RGB').save(save_path, format='PNG', quality=100, optimize=False)
        print(f"Saved colorized image to: {save_path}")
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def colorize_images(target_path, colorizator, path, size, denoiser, denoiser_sigma):
    images = os.listdir(path)
    for image_path in images:
        file_path = os.path.join(path, image_path)
        if os.path.isdir(file_path):
            continue
        image_name, image_ext = os.path.splitext(image_path)
        if image_ext.lower() not in ('.jpg', '.jpeg'):
            image_path = image_name + '.jpg'
        print(f'Processing: {file_path}')
        save_path = os.path.join(target_path, image_path)
        colorize_single_image(file_path, save_path, colorizator, size, denoiser, denoiser_sigma)

def create_colorizer(device, generator, denoiser):
    return MangaColorizator(device, generator, denoiser)

def parse_args():
    parser = argparse.ArgumentParser(description="""
        Colorizes manga images using AI.
        
        Examples:
            # Colorize single image:
            python inference_v2.py -p manga.png -g
            
            # Colorize with 2x zoom:
            python inference_v2.py -p manga.png -z 2
            
            # Colorize folder using GPU and higher resolution:
            python inference_v2.py -p ./manga_folder -g -s 832
            
            # Colorize without denoiser and 1.5x zoom:
            python inference_v2.py -p manga.png -nd -z 1.5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-p", "--path", required=True, help="Path to the image or directory")
    parser.add_argument("-gen", "--generator", default='networks/generator.zip', help="Path to the generator model")
    parser.add_argument("-des_path", "--denoiser_path", default='denoising/models/', help="Path to the denoiser model")
    parser.add_argument("-s", "--save_path", default=None, help="Custom path to save colorized images")
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true', help="Force usage of GPU")
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false', help="Disable denoising")
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25, help="Denoiser sigma value")
    parser.add_argument("-sz", "--size", type=int, default=576, help="Size for the colorization process")
    parser.add_argument("-z", "--zoom", type=float, default=1.0,
                      help="Zoom factor for final image (e.g., 1.5, 2, 2.5). Default: 1.0")
    parser.add_argument("-ai", "--ai_upscale", action='store_true',
                      help="Use AI for upscaling (slower but better quality)")
    
    args = parser.parse_args()
    
    if args.size % 32 != 0:
        parser.error("Size parameter must be multiple of 32")
    
    if args.zoom < 0.1 or args.zoom > 4.0:
        parser.error("Zoom must be between 0.1 and 4.0")
    
    return args

def main():
    args = parse_args()

    # Set the device: use GPU if available and not forced to use CPU
    device = 'cuda' if args.gpu or (torch.cuda.is_available() and not args.gpu) else 'cpu'
    
    colorizer = create_colorizer(device, args.generator, args.denoiser_path)
    
    if os.path.isdir(args.path):
        # Create a "colorization" folder in the specified path if no custom save path is provided
        colorization_path = args.save_path if args.save_path else os.path.join(args.path, 'colorization')
        os.makedirs(colorization_path, exist_ok=True)
        colorize_images(colorization_path, colorizer, args.path, args.size, args.denoiser, args.denoiser_sigma)
        
    elif os.path.isfile(args.path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        directory_path = os.path.dirname(args.path)
        image_name_with_ext = os.path.basename(args.path)
        image_name, image_ext = os.path.splitext(image_name_with_ext)
        
        if image_ext.lower() in image_extensions:
            zoom_info = f'_x{args.zoom}' if args.zoom != 1.0 else ''
            template_name = f"{image_name}_colorized{zoom_info}.png"
            
            new_image_path = os.path.join(args.save_path if args.save_path else directory_path, template_name)
            colorize_single_image(args.path, new_image_path, colorizer, args.size, args.denoiser, 
                                args.denoiser_sigma, args.zoom, args.gpu, args.ai_upscale)
        else:
            print('Wrong image format')
    else:
        print('Wrong path format')

if __name__ == "__main__":
    main()
