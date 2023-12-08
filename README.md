## Установка

1. Убедитесь, что у вас установлен Python версии 3.9.13:
   ```bash
    cd "C:\Users\<username>\AppData\Local\Programs\Python\Python39"
    python --version
    ```
2. Перейдите к папке для хранения проектов (добавьте ваш путь до папки с проектами):
   ```bash
    cd "C:\Users\<username>\<PythonProjects>"
    ```
3. Склонируйте репозиторий на локальную машину (добавьте ваше название папки для проекта):
    ```bash
    git clone -b flask_service_nlp https://gitlab.com/Pavel_Demukhametov/projecthub "<Project>"
    ```
4. Перейдите к папке с клонированным проектом:
   ```bash
    cd "<Project>"
    ```
5. Создайте виртуального окружения с версией Python 3.9.13
   ```bash
    "C:\Users\<username>\AppData\Local\Programs\Python\Python39\python.exe" -m venv "myenv"
    ```
6. Активируйте виртуальное окружение:
   ```bash
   "myenv"\Scripts\activate
   ```
7. Установите зависимости из requirements.txt:
   ```bash
    pip install -r requirements.txt
    ```
8. Запуск приложения
   ```bash
    python app.py
    ```