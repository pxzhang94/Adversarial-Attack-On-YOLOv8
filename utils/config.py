import sys
sys.path.append("../")

from tutorial.transform_function import *

BRIGHTNESS_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]  # 亮度系数（<1变暗，>1变亮）

COMPRESSION_FACTORS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

CONTRAST_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

FGSM_SIZE = [0.033, 0.067, 0.1, 0.133, 0.167, 0.2, 0.233, 0.267, 0.3]

GAUSSIAN_BLUR_SIGMAS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]

GAUSSIAN_NOISE_SIGMAS = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

HUE_FACTORS = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]

ROTATION_ANGLES = [-20, -16, -12, -8, -4, 4, 8, 12, 16, 20]

SATURATION_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

TRANSFORM_CONFIG = {
    "brightness": BRIGHTNESS_FACTORS,
    "compression": COMPRESSION_FACTORS,
    "contrast": CONTRAST_FACTORS,
    "fgsm": FGSM_SIZE,
    "gaussian_blur": GAUSSIAN_BLUR_SIGMAS,
    "gaussian_noise": GAUSSIAN_NOISE_SIGMAS,
    "hue": HUE_FACTORS,
    "rotation": ROTATION_ANGLES,
    "saturation": SATURATION_FACTORS
}

TRANSFORM_FUNCTION = {
    # "brightness": apply_brightness,
    # "compression": apply_jpeg_compression,
    # "contrast": apply_contrast,
    "fgsm": apply_fgsm,
    # "gaussian_blur": apply_gaussian_blur,
    # "gaussian_noise": apply_gaussian_noise,
    # "hue": apply_hue,
    # "rotation": apply_rotation,
    # "saturation": apply_saturation
}

