import nltk
from flask import Flask, request, jsonify, Response
from package_NLP import dock_processing

app = Flask(__name__)

# Проверка установки необходимых пакетов
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@app.route('/data', methods=['GET'])
def get_data():
    url = request.args.get('url')

    if not url:
        return jsonify({'error': 'Missing URL parameter'}), 400

    try:
        json_dock_data = dock_processing(url)
        # Возвращаем данные в ответе Flask
        return Response(json_dock_data, content_type='application/json; charset=utf-8')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
