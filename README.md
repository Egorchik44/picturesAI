
# Image Processing API

Этот проект представляет собой RESTful API для обработки изображений с использованием Flask, TensorFlow и PyTorch. API предоставляет возможность колоризации изображений, удаления бликов и улучшения качества изображений.

## Оглавление

- [Установка](#установка)
- [Использование](#использование)
- [Конечные точки API](#конечные-точки-api)
- [Тестирование](#тестирование)

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ваш-пользователь/ваш-репозиторий.git
   cd ваш-репозиторий
   ```

2. Создайте виртуальное окружение и активируйте его:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Для Linux/MacOS
   venv\Scripts\activate  # Для Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Запустите сервер:
   ```bash
   python app.py
   ```

## Использование

После запуска сервера, API будет доступен по адресу `http://127.0.0.1:5000`. Вы можете использовать Postman или любой другой HTTP-клиент для отправки запросов к API.

## Конечные точки API

### Колоризация изображения

**POST** `/colorize`

Отправьте изображение в формате `multipart/form-data` с ключом `photo`.

Пример запроса:
```bash
curl -X POST -F "photo=@path/to/your/image.jpg" http://127.0.0.1:5000/colorize
```

Пример ответа:
```json
{
  "colorized_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
}
```

### Удаление бликов

**POST** `/remove_glare`

Отправьте изображение в формате `multipart/form-data` с ключом `photo`.

Пример запроса:
```bash
curl -X POST -F "photo=@path/to/your/image.jpg" http://127.0.0.1:5000/remove_glare
```

Пример ответа:
```json
{
  "glare_removed_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
}
```

### Улучшение качества изображения

**POST** `/enhance_quality`

Отправьте изображение в формате `multipart/form-data` с ключом `photo`.

Пример запроса:
```bash
curl -X POST -F "photo=@path/to/your/image.jpg" http://127.0.0.1:5000/enhance_quality
```

Пример ответа:
```json
{
  "enhanced_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
}
```

## Тестирование

Для запуска тестов используйте следующую команду:
```bash
python -m unittest discover tests
```
