from flask import current_app as app, jsonify

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello, World!")
