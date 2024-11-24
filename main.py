import subprocess
import os

inference_script = "manga-colorization-v2-custom/inference_v2.py"
# inference_script = "inference.py" # need to cd to the directory containing this script
image_path = "magi_functional/data_test/personal_data/Ruri_Dragon/raw/"
denoiser_sigma = 0
use_gpu = True
save_path = "./test_output_coloring"
generator_path = "manga-colorization-v2-custom/networks/generator.zip"
denoiser_path = "manga-colorization-v2-custom/denoising/models/net_rgb.pth"


required_paths = [image_path, generator_path, denoiser_path]

if all(os.path.exists(path) for path in required_paths):
    # If all required paths exist, run the subprocess
    subprocess.run([
        'python', inference_script, 
        '-p', image_path, 
        '-des_path', denoiser_path, 
        '-gen', generator_path, 
        '-s', save_path, 
        '-ds', str(denoiser_sigma)
    ] + (['--gpu'] if use_gpu else []))
else:
    print("One or more required paths do not exist. Please check the following:")
