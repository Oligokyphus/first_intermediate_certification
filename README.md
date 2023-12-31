# Веб-приложение для анализа данных и проверки гипотез

Этот репозиторий содержит веб-приложение, которое позволяет загружать CSV-файлы в качестве датасетов, визуализировать распределение данных по выбранным столбцам и выполнять два различных алгоритма для проверки гипотез. Данное веб-приложение было разработано в рамках домашнего задания по темам "Статистика", "Визуализация" и "Развертывание в виде веб-приложения".

## Инструкции

1. Установите необходимые библиотеки, выполнив следующую команду:

        pip install -r requirements.txt

2. Запустите веб-приложение, используя следующую команду:

        streamlit run app.py

3. После запуска приложения, вы увидите интерфейс, где можно загрузить CSV-файл с данными.

4. Выберите две переменные для анализа из загруженных данных.

5. Выберите один из двух алгоритмов для проверки гипотез: "t-test" для некатегориальных данных или "chi-square test" для категориальных данных.

6. После выбора алгоритма, результаты проверки гипотез и интерпретация будут выведены на странице.

## Файлы

* app.py: Исходный код веб-приложения.
* requirements.txt: Список необходимых библиотек для корректной работы приложения.
* titanic.csv: Датасет для тестирования приложения. 

## Зависимости

Для работы этого приложения требуются следующие библиотеки:

* pandas
* streamlit
* bokeh
* numpy
* scipy

## Как использовать

1. Загрузите CSV-файл с данными.
2. Выберите две переменные для анализа.
3. Выберите один из алгоритмов для проверки гипотез: "t-test" или "chi-square test".
4. Получите результаты проверки гипотез и их интерпретацию.
