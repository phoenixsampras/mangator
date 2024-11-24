import torch
from torchvision.transforms import ToTensor
import numpy as np

from networks.models import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad, resize_image, undo_padding

class MangaColorizator:
    def __init__(self, device, generator_path='networks/generator.zip', denoiser_path='denoising/models/'):
        self.device = device
        self.colorizer = Colorizer().to(device)
        self.load_generator(generator_path)
        self.denoiser = FFDNetDenoiser(device=device, weights_dir=denoiser_path)

        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        self.original_size = None

    def load_generator(self, generator_path):
        """Load the generator model's state dictionary."""
        state_dict = torch.load(generator_path, map_location=self.device, weights_only=True)
        self.colorizer.generator.load_state_dict(state_dict)
        self.colorizer.eval()  # Set the colorizer to evaluation mode

    def set_image(self, image, size=576, apply_denoise=True, denoise_sigma=25, transform=ToTensor()):
        if size % 32 != 0:
            raise RuntimeError("Size must be divisible by 32")

        self.original_size = (image.shape[0], image.shape[1])

        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)

        image, self.current_pad = resize_pad(image, size)
        
        self.current_image = transform(image).unsqueeze(0).to(self.device)
        
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3]).to(self.device)

    def update_hint(self, hint, mask):
        """
        Update the hint for the colorization process.

        Args:
            hint: numpy.ndarray with shape (H, W, 3)
            mask: numpy.ndarray with shape (H, W)
        """
        hint = hint.astype('float32') / 255 if np.issubdtype(hint.dtype, np.integer) else hint

        hint = (hint - 0.5) / 0.5  # Normalize the hint
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def colorize(self):
        """Perform colorization on the current image."""
        if self.current_image is None:
            raise RuntimeError("No image has been set. Call set_image() first.")

        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
            fake_color = fake_color.detach()

        result = fake_color[0].cpu().permute(1, 2, 0).numpy()
        result = (result * 0.5 + 0.5)  # Rescale to [0, 1]

        # Remove padding
        result = undo_padding(result, self.current_pad)

        # Resize back to original dimensions if we stored them
        if self.original_size is not None:
            original_height, original_width = self.original_size
            result = resize_image(result, (original_width, original_height))

        return result


