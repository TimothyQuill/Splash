"""
REST API definition.
"""
from flask import Flask, jsonify, request
from generate import generate_sequence


app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def automated():
    """
        Endpoint to generate a sequence based on the provided input data.

        This function handles POST requests, extracts JSON data from the request body,
        and uses the `generate_sequence` function to create a sequence. The generated
        sequence is then returned as a JSON response.

        Args:
            None (Inputs are provided in the POST request body as JSON).

        Returns:
            Response (flask.Response): A JSON response containing the generated sequence.
        """
    request_json = request.get_json(silent=True)
    sequence = generate_sequence(request_json)
    return jsonify(sequence)


if __name__ == '__main__':
    app.run(debug=True)
