from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import cv2
import torch
import base64
import numpy as np
from torchvision import transforms
from PIL import Image
from colorized import get_model
from image_utils import split_image, merge_tiles, preprocess_tile, process_tile, allowed_file
from tensorflow.keras.models import load_model

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

SHARPEN_MODEL_PATH = r"C:\Users\Егор\Downloads\ai-quasar-git\model\photo_sharpen_model.h5"

# Обработка запроса на колоризацию
@app.route('/colorize', methods=['POST'])
def colorize():
    """Endpoint для колоризации изображения."""
    logger.info("Получен запрос на колоризацию")

    try:
        if 'photo' not in request.files:
            logger.warning("Изображение не найдено в запросе")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['photo']
        logger.info(f"Получен файл: {file.filename}")

        if file.filename == '':
            logger.warning("Файл не имеет имени")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            logger.warning("Недопустимый тип файла")
            return jsonify({'error': 'Invalid file type'}), 400

        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            logger.error("Не удалось декодировать изображение")
            return jsonify({'error': 'Failed to decode image'}), 400

        logger.info(f"Размер загруженного изображения: {img.shape}")

        # Получаем модель
        model = get_model()

        # Обработка изображения
        tiles, positions, original_size = split_image(img)
        logger.info(f"Изображение разделено на {len(tiles)} тайлов")

        processed_tiles = []
        for idx, tile in enumerate(tiles):
            logger.debug(f"Обработка тайла {idx + 1}/{len(tiles)}")
            img_light, img_input = preprocess_tile(tile)
            with torch.no_grad():
                output = model(img_input)
            processed_tile = process_tile(output.numpy(), img_light)
            processed_tiles.append(processed_tile)

        logger.info("Все тайлы обработаны, сборка финального изображения")
        final_image = merge_tiles(processed_tiles, positions, original_size)

        # Конвертация результата в base64
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", final_image_bgr)

        if not is_success:
            logger.error("Не удалось закодировать результат в PNG")
            return jsonify({'error': 'Failed to encode image'}), 500

        img_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.info("Изображение успешно обработано и закодировано")

        return jsonify({'colorized_image': f'data:image/png;base64,{img_base64}'})

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# Обработка запроса на удаление бликов
@app.route('/remove_glare', methods=['POST'])
def remove_glare():
    logger.info("Получен запрос на удаление бликов")
    
    try:
        if 'photo' not in request.files:
            logger.warning("Изображение не найдено в запросе")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['photo']
        logger.info(f"Получен файл: {file.filename}")
        
        if file.filename == '':
            logger.warning("Файл не имеет имени")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            logger.warning("Недопустимый тип файла")
            return jsonify({'error': 'Invalid file type'}), 400

        # Read and convert image
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            logger.error("Не удалось декодировать изображение")
            return jsonify({'error': 'Failed to decode image'}), 400

        logger.info(f"Размер загруженного изображения: {img.shape}")
        
        # Получаем модель
        model = get_model(model_type='unet')

        # Обработка изображения
        tiles, positions, original_size = split_image(img)
        logger.info(f"Изображение разделено на {len(tiles)} тайлов")

        processed_tiles = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for idx, tile in enumerate(tiles):
            logger.debug(f"Обработка тайла {idx+1}/{len(tiles)}")
            tile_pil = Image.fromarray(tile)
            input_tensor = transform(tile_pil).unsqueeze(0)
            
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # Convert back to numpy array
            processed_tile = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
            processed_tile = (processed_tile * 255).astype(np.uint8)
            processed_tiles.append(processed_tile)
        
        logger.info("Все тайлы обработаны, сборка финального изображения")
        final_image = merge_tiles(processed_tiles, positions, original_size)
        
        # Convert to BGR for cv2 encoding
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", final_image_bgr)
        
        if not is_success:
            logger.error("Не удалось закодировать результат в PNG")
            return jsonify({'error': 'Failed to encode image'}), 500
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.info("Изображение успешно обработано и закодировано")
        
        return jsonify({'glare_removed_image': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/enhance_quality', methods=['POST'])
def enhance_quality():
    logger.info("Получен запрос на улучшение качества")
    
    try:
        if 'photo' not in request.files:
            logger.warning("Изображение не найдено в запросе")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['photo']
        logger.info(f"Получен файл: {file.filename}")
        
        if file.filename == '':
            logger.warning("Файл не имеет имени")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            logger.warning("Недопустимый тип файла")
            return jsonify({'error': 'Invalid file type'}), 400

        # Read and convert image
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            logger.error("Не удалось декодировать изображение")
            return jsonify({'error': 'Failed to decode image'}), 400

        logger.info(f"Размер загруженного изображения: {img.shape}")
        
        # Загрузка модели
        model = load_model(SHARPEN_MODEL_PATH)

        # Обработка изображения
        tiles, positions, original_size = split_image(img)
        logger.info(f"Изображение разделено на {len(tiles)} тайлов")

        processed_tiles = []
        for idx, tile in enumerate(tiles):
            logger.debug(f"Обработка тайла {idx+1}/{len(tiles)}")
            tile_pil = Image.fromarray(tile)
            tile_tensor = transforms.ToTensor()(tile_pil).unsqueeze(0)
            
            # Преобразование тензора PyTorch в массив NumPy и изменение формы
            tile_array = tile_tensor.numpy().transpose((0, 2, 3, 1))  # (1, 256, 256, 3)
            
            with torch.no_grad():
                output_tensor = model(tile_array)
            
            # Преобразование output_tensor в массив NumPy и изменение формы
            processed_tile = output_tensor.numpy().squeeze(0)  # (256, 256, 3)
            processed_tile = (processed_tile * 255).astype(np.uint8)
            processed_tiles.append(processed_tile)
        
        logger.info("Все тайлы обработаны, сборка финального изображения")
        final_image = merge_tiles(processed_tiles, positions, original_size)
        
        # Convert to BGR for cv2 encoding
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", final_image_bgr)
        
        if not is_success:
            logger.error("Не удалось закодировать результат в PNG")
            return jsonify({'error': 'Failed to encode image'}), 500
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.info("Изображение успешно обработано и закодировано")
        
        return jsonify({'enhanced_image': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Запуск сервера")
    app.run(debug=True)