from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load your pre-trained model here
with open('HTNDoodl.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/', methods=['GET'])
def hello_world():
    """Basic route"""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def receive_post_request():
    try:
        # Get JSON data from the request
        received_data = request.get_json()

        # Assuming the array is sent as 'your_array' key in the JSON data
        your_array = received_data.get('your_array', None)

        # Perform operations with your_array using the loaded model
        prediction = model.predict([your_array])

        # Return the prediction or any other response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Handle exceptions appropriately
        print('Error:', str(e))
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
