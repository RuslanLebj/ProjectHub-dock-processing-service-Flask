from flask import Blueprint, request, jsonify, Response
from document_processing.processor import process_document

bp = Blueprint('main', __name__)


@bp.route('/data', methods=['GET'])
def get_data():
    url = request.args.get('url')

    if not url:
        return jsonify({'error': 'Missing URL parameter'}), 400

    try:
        json_dock_data = process_document(url)
        return Response(json_dock_data, content_type='application/json; charset=utf-8')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
