from flask import Flask
from package_NLP import dock_processing

app = Flask(__name__)


@app.route('/data', methods=['GET'])
def get_data():
    if request.method == 'POST':
        # Получаем url адрес документа
        url_dock_address = request.json.get('url')
        # Обраюатываем документ по переданному адресу, получаем json с информацией
        json_dock_data = dock_processing(url_dock_address)
        return json_dock_data


if __name__ == '__main__':
    app.run(debug=True)
