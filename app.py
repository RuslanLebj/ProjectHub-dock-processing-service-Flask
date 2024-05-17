from flask import Flask
from routes import bp as routes_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    app.register_blueprint(routes_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
