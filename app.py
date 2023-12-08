from flask import Flask, request
from package_NLP import dock_processing

app = Flask(__name__)


@app.route('/data', methods=['GET'])
def get_data():
    url = request.args.get('url')

    if not url:
        return jsonify({'error': 'Missing URL parameter'}), 400

    try:
        json_dock_data = dock_processing(url)
        return json_dock_data
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
