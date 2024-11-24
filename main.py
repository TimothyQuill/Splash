from flask import Flask, jsonify, request
from generate import generate_sequence


app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def automated():
    request_json = request.get_json(silent=True)
    sequence = generate_sequence(request_json)
    print('sequence', sequence)
    return jsonify(sequence)


if __name__ == '__main__':
    app.run(debug=True)
