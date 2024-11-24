import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torch.nn import functional as F

MODEL_PATH = 'models/RealESRGAN_x4plus_anime_6B.pth'

def check_cuda():
    try:
        return torch.cuda.is_available() and hasattr(torch.cuda, 'is_available') and torch.cuda.is_initialized()
    except (AssertionError, RuntimeError):
        return False

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, padding=1)
        
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        
        self.conv_body = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        body_fea = self.body(fea)
        trunk = self.conv_body(body_fea)
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_hr(fea))
        out = self.conv_last(fea)
        return out

class AIUpscaler:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if check_cuda() else 'cpu'
        self.device = device
        print(f"AIUpscaler: Usando {self.device} para procesamiento")
        self.model = None
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró el modelo en {MODEL_PATH}. "
                "Por favor, asegúrate de tener el archivo RealESRGAN_x4plus_anime_6B.pth en la carpeta models/"
            )
        
    def load_model(self, scale_factor=4):
        if self.model is None:
            print("Cargando modelo RealESRGAN optimizado para anime...")
            self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=6)
            
            try:
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                if 'params_ema' in state_dict:
                    self.model.load_state_dict(state_dict['params_ema'])
                else:
                    self.model.load_state_dict(state_dict)
                print("Modelo cargado exitosamente!")
            except Exception as e:
                raise RuntimeError(f"Error cargando el modelo: {str(e)}")
                
            self.model.eval()
            self.model = self.model.to(self.device)
    
    def upscale(self, image, scale_factor=4):
        """
        Aumenta la resolución de una imagen usando RealESRGAN
        """
        self.load_model(scale_factor)
        
        # Convertir imagen a tensor
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preparar input
        img = np.array(image)
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        try:
            with torch.no_grad():
                img = img.to(self.device)
                output = self.model(img)
                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "Error: Memoria GPU insuficiente. "
                    "Intenta con una imagen más pequeña o usa CPU con -ai sin -g"
                )
            raise e
        
        # Post-procesar
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        
        # Ajustar al factor de escala deseado si es diferente de 4
        if scale_factor != 4:
            print(f"Ajustando escala de x4 a x{scale_factor}")
            h, w = output.shape[:2]
            target_h = int(h * (scale_factor / 4))
            target_w = int(w * (scale_factor / 4))
            output = Image.fromarray(output).resize((target_w, target_h), Image.Resampling.LANCZOS)
            output = np.array(output)
            
        return output

def upscale_image(image, scale_factor=2, device=None):
    if device == 'cuda' and not check_cuda():
        print("Warning: GPU solicitada pero CUDA no disponible. Usando CPU.")
        device = 'cpu'
    upscaler = AIUpscaler(device)
    return upscaler.upscale(image, scale_factor)