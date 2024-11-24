# Manga PDF Colorizator

Este módulo permite colorizar PDFs de manga en blanco y negro, manteniendo las páginas que ya están en color.

## Requisitos

- Python 3.7+
- PyMuPDF (fitz)
- NumPy
- Pillow
- OpenCV
- PyTorch

## Instalación

```

## Opciones

- `-o, --output`: Ruta de salida para el PDF
- `-g, --gpu`: Usar GPU para el procesamiento
- `-s, --size`: Tamaño de procesamiento (default: 576)
- `-nd, --no_denoise`: Desactivar denoiser
- `-ds, --denoiser_sigma`: Intensidad del denoiser (default: 25)
- `-z, --zoom`: Factor de zoom (default: 1.0)
- `-ai, --ai_upscale`: Usar IA para upscaling

## Notas

- El módulo detecta automáticamente páginas en color y las mantiene sin procesar
- El tamaño de procesamiento debe ser múltiplo de 32
- El factor de zoom debe estar entre 0.1 y 4.0