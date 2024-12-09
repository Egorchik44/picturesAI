
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const resultSection = document.getElementById('result-section');
    const originalPhoto = document.getElementById('original-photo');
    const enhancedPhoto = document.getElementById('unsharpen-photo');
    const uploadButton = document.querySelector('.submit-button');
    const downloadBtn = document.getElementById('download-btn');

    // Функция для включения кнопки скачивания
    function enableDownloadButton() {
        downloadBtn.disabled = false;
    }

    // Функция скачивания изображения
    function downloadEnhancedImage() {
        const imageData = enhancedPhoto.src;
        const link = document.createElement('a');
        
        link.href = imageData;
        link.download = 'enhanced_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Обработчик события для кнопки скачивания
    downloadBtn.addEventListener('click', downloadEnhancedImage);

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const file = formData.get('photo');
        
        if (!file) {
            alert('Пожалуйста, выберите файл для загрузки');
            return;
        }

        try {
            // Show loading state
            uploadButton.disabled = true;
            uploadButton.textContent = 'Обработка...';

            // Display original photo immediately
            const originalUrl = URL.createObjectURL(file);
            originalPhoto.src = originalUrl;

            // Send image to backend for processing
            const response = await fetch('http://localhost:5000/enhance_quality', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Display enhanced image
            enhancedPhoto.src = data.enhanced_image;
            resultSection.style.display = 'block';

            // Включаем кнопку скачивания после успешной обработки
            enableDownloadButton();

        } catch (error) {
            console.error('Error:', error);
            alert('Произошла ошибка при обработке изображения: ' + error.message);
        } finally {
            // Reset button state
            uploadButton.disabled = false;
            uploadButton.textContent = 'Повысить качество';
        }
    });

    // Add file input change handler to show selected filename
    const fileInput = document.getElementById('photo-upload');
    fileInput.addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name;
        const label = document.querySelector('.upload-button');
        label.textContent = fileName || 'Выберите файл';
    });

    // Дополнительный код для работы со слайдером
    const slider = document.getElementById('comparison-slider');
    const sliderButton = document.querySelector('.slider-button');

    slider.addEventListener('input', function() {
        const sliderValue = this.value;
        enhancedPhoto.style.clipPath = `polygon(${sliderValue}% 0, 100% 0, 100% 100%, ${sliderValue}% 100%)`;
        sliderButton.style.left = `calc(${sliderValue}% - 18px)`;
    });
});