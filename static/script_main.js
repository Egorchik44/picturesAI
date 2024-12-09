const themeToggleButton = document.getElementById('theme-toggle');
const body = document.body;


// Функция для установки темы
function setTheme(themeName) {
    localStorage.setItem('theme', themeName);
    document.documentElement.className = themeName;
}

// Функция для переключения темы
function toggleTheme() {
    if (localStorage.getItem('theme') === 'dark-theme') {
        setTheme('light-theme');
    } else {
        setTheme('dark-theme');
    }
}

// Обработчик события для кнопки переключения темы
document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('theme-toggle').addEventListener('click', function() {
        toggleTheme();
        if (localStorage.getItem('theme') === 'dark-theme') {
            this.textContent = 'Светлая тема';
        } else {
            this.textContent = 'Темная тема';
        }
    });
});