# Manga PDF Colorizator 🎨

An advanced AI-powered tool for colorizing black and white manga PDFs while preserving existing colored pages and text quality.

## 🌟 Features

- **Smart Colorization**: Automatically detects and preserves existing colored pages
- **Text Preservation**: Enhanced text clarity and sharpness in speech bubbles
- **Border Optimization**: Automatic white border removal for cleaner output
- **AI Upscaling**: Optional RealESRGAN upscaling for higher quality
- **GPU Acceleration**: CUDA support for faster processing

## 📋 Requirements

### System Requirements
- Python 3.7+
- CUDA-capable GPU (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

### Dependencies
```bash
# Core dependencies
PyMuPDF==1.21.1
numpy>=1.21.0
Pillow>=9.0.0
opencv-python>=4.5.0
torch>=1.9.0
colorama>=0.4.4
basicsr>=1.4.2  # For RealESRGAN
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ahorasoft/manga-colorizator.git
cd manga-colorizator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
# Create directories
mkdir -p networks denoising/models

# Download colorization model
wget https://github.com/ahorasoft/manga-colorizator/releases/download/v1.0/generator.zip -O networks/generator.zip

# Download denoising models
wget https://github.com/ahorasoft/manga-colorizator/releases/download/v1.0/denoising_models.zip
unzip denoising_models.zip -d denoising/models/

# Download RealESRGAN model (optional, for upscaling)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth -P upscaling/
```

## 💻 Usage Examples

### Command Line Usage
```bash
# Basic usage
python inference_pdf.py "manga.pdf"

# Advanced usage with all options
python inference_pdf.py \
    -sp 1 -ep 140 \        # Start page 1, end page 140
    -z 1.5 \               # Zoom factor 1.5x
    -ai \                  # Enable AI upscaling
    -pt 0 \               # Disable text preservation
    -ci 1 \               # Normal color intensity
    "path/to/manga.pdf"

# Real example
python inference_pdf.py -sp 1 -ep 140 -z 1.5 -ai -pt 0 -ci 1 "manga/nakidpics.pdf"

# Process specific chapter with GPU
python inference_pdf.py \
    -sp 45 -ep 65 \       # Process pages 45-65 only
    -g \                  # Enable GPU
    -s 768 \              # Higher resolution
    -ds 25 \              # Denoising strength
    "manga/chapter5.pdf"
```

Available options:
- `-sp, --start_page`: Start page number
- `-ep, --end_page`: End page number
- `-z, --zoom`: Output zoom factor (0.1-4.0)
- `-ai, --ai_upscale`: Enable RealESRGAN upscaling
- `-pt, --preserve_text`: Text preservation (0.0-1.0)
- `-ci, --color_intensity`: Color intensity (0.5-2.0)
- `-g, --gpu`: Enable GPU acceleration
- `-s, --size`: Processing size (multiple of 32)
- `-ds, --denoiser_sigma`: Denoising strength (1-50)

## 🛠️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `input_colorized.pdf` | Output PDF path |
| `-g, --gpu` | `False` | Enable GPU acceleration |
| `-s, --size` | `576` | Processing size (must be multiple of 32) |
| `-nd, --no_denoise` | `False` | Disable denoising |
| `-ds, --denoiser_sigma` | `25` | Denoising strength (1-50) |
| `-z, --zoom` | `1.0` | Output zoom factor (0.1-4.0) |
| `-ai, --ai_upscale` | `False` | Enable RealESRGAN upscaling |
| `-pt, --preserve_text` | `0.5` | Text preservation strength (0.0-1.0) |
| `-ci, --color_intensity` | `1.0` | Color intensity (0.5-2.0) |

## 📁 Project Structure
```
manga-colorizator/
├── networks/
│   └── generator.zip          # Colorization model
├── denoising/
│   └── models/               # Denoising models
├── upscaling/
│   └── RealESRGAN_x4plus_anime_6B.pth  # Upscaling model
├── inference.py              # Single image colorization
├── inference_pdf.py          # PDF processing
└── colorize.py              # CLI interface
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Could not find generator.zip
   Solution: Ensure all models are downloaded to correct directories
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce processing size
   colorizer = PDFColorizer(
       use_gpu=True,
       size=384  # Default is 576
   )
   ```

3. **Blurry Text**
   ```python
   # Increase text preservation
   colorizer = PDFColorizer(
       preserve_text=0.8,
       denoiser_sigma=15
   )
   ```

## 📧 Contact

- Website: http://www.ahorasoft.com
- Email: support@ahorasoft.com
- Issues: [GitHub Issues](https://github.com/ahorasoft/manga-colorizator/issues)

---
Made with ❤️ by Ahorasoft