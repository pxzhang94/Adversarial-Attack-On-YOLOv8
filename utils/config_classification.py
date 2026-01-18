import sys
sys.path.append("../")

from tutorial.transform_function import *

BRIGHTNESS_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]  # 亮度系数（<1变暗，>1变亮）

COMPRESSION_FACTORS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

CONTRAST_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

FGSM_SIZE = [0.006, 0.012, 0.018, 0.024, 0.03, 0.036, 0.042, 0.048, 0.054, 0.06]

PGD_SIZE = [0.006, 0.012, 0.018, 0.024, 0.03, 0.036, 0.042, 0.048, 0.054, 0.06]

CW_KAPPA = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

GAUSSIAN_BLUR_SIGMAS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]

DEFOCUS_BLUR_STRENGTH = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

MOTION_BLUR_SHIFT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

GAUSSIAN_NOISE_SIGMAS = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

SALT_PEPPER_NOISE_RATIO = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

POISSON_NOISE_SCALE = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

HUE_FACTORS = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]

ROTATION_ANGLES = [-20, -16, -12, -8, -4, 4, 8, 12, 16, 20]

SATURATION_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

TOG_VANISHING = [
    {"n_iter": 3,  "eps": 4,  "eps_iter": 1},
    {"n_iter": 5,  "eps": 4,  "eps_iter": 1},
    {"n_iter": 10, "eps": 4,  "eps_iter": 1},

    {"n_iter": 5,  "eps": 8,  "eps_iter": 2},
    {"n_iter": 10, "eps": 8,  "eps_iter": 1},
    {"n_iter": 10, "eps": 8,  "eps_iter": 2},

    {"n_iter": 15, "eps": 12, "eps_iter": 2},
    {"n_iter": 20, "eps": 12, "eps_iter": 1},

    {"n_iter": 20, "eps": 16, "eps_iter": 2},
    {"n_iter": 5,  "eps": 16, "eps_iter": 4},
]


TRANSFORM_CONFIG = {
    "brightness": BRIGHTNESS_FACTORS,
    "compression": COMPRESSION_FACTORS,
    "contrast": CONTRAST_FACTORS,
    "fgsm": FGSM_SIZE,
    "gaussian_blur": GAUSSIAN_BLUR_SIGMAS,
    "gaussian_noise": GAUSSIAN_NOISE_SIGMAS,
    "hue": HUE_FACTORS,
    "rotation": ROTATION_ANGLES,
    "saturation": SATURATION_FACTORS,
    "defocus_blur": DEFOCUS_BLUR_STRENGTH,
    "motion_blur": MOTION_BLUR_SHIFT,
    "salt_pepper_noise": SALT_PEPPER_NOISE_RATIO,
    "poisson_noise": POISSON_NOISE_SCALE,
    "pgd": PGD_SIZE,
    "cw": CW_KAPPA,
    "tog_vanishing": TOG_VANISHING,
}

TRANSFORM_FUNCTION = {
    "brightness": apply_brightness,
    # "compression": apply_jpeg_compression,
    "contrast": apply_contrast,
    "fgsm": apply_fgsm,
    "cw": apply_cw,
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    # "hue": apply_hue,
    # "rotation": apply_rotation,
    "saturation": apply_saturation,
    'defocus_blur': apply_defocus_blur,
    "motion_blur": apply_motion_blur,
    "salt_pepper_noise": apply_salt_pepper_noise,
    "poisson_noise": apply_poisson_noise,
    "pgd": apply_pgd,
}

