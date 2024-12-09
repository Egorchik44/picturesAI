import numpy as np
import math
from skimage import color, transform
import torch

def split_image(image, tile_size=256):
    h, w = image.shape[:2]
    tiles, positions = [], []
    n_h = math.ceil(h / tile_size)
    n_w = math.ceil(w / tile_size)
    pad_h, pad_w = n_h * tile_size - h, n_w * tile_size - w
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    for i in range(n_h):
        for j in range(n_w):
            y1, y2 = i * tile_size, (i + 1) * tile_size
            x1, x2 = j * tile_size, (j + 1) * tile_size
            tiles.append(padded_image[y1:y2, x1:x2])
            positions.append((y1, x1))
    return tiles, positions, (h, w)

def merge_tiles(tiles, positions, original_size, tile_size=256):
    h, w = original_size
    merged = np.zeros((math.ceil(h / tile_size) * tile_size, math.ceil(w / tile_size) * tile_size, 3), dtype=np.uint8)
    for tile, (y, x) in zip(tiles, positions):
        merged[y:y+tile_size, x:x+tile_size] = tile
    return merged[:h, :w]

def preprocess_tile(tile):
    img_lab = color.rgb2lab(tile)
    img_light = img_lab[:, :, 0]
    img_input = torch.from_numpy(img_light[None, None, :, :]).float()
    return img_light, img_input

def process_tile(output, img_light):
    img = output.squeeze().transpose(1, 2, 0)
    img_lab = np.insert(img, 0, img_light, axis=2)
    img_rgb = (color.lab2rgb(img_lab) * 255).astype(np.uint8)
    return img_rgb

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS