from flask import Flask, request, send_file, render_template, jsonify, url_for
import pandas as pd
import pickle
import io
import os
from werkzeug.utils import secure_filename
import openai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['OPENAI_API_KEY'] = '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'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your model
model_path = 'models/logistic_regression_model.pkl'
loaded_model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

def handle_missing_values(data):
    # Your existing function to handle missing values
    data['HIGH EQUITY'] = data['HIGH EQUITY'].fillna(0)
    data['INTER FAMILY TRANSFER'] = data['INTER FAMILY TRANSFER'].fillna(0)
    data['55+'] = data['55+'].fillna(0)
    data['TAXES'] = data['TAXES'].fillna(0)
    data = data.fillna(0)
    return data

def process_file(file_path):
    # Your existing function to process the file
    df = pd.read_excel(file_path)
    tag_columns = ['FOLIO', 'ABSENTEE', 'HIGH EQUITY', 'DOWNSIZING', 'VACANT', '55+', 'INTER FAMILY TRANSFER', 'TAXES']
    df = df[tag_columns]
    df = df.drop_duplicates('FOLIO')
    df = handle_missing_values(df)
    data = df.drop('FOLIO', axis=1)
    predictions = loaded_model.predict_proba(data)[:, 1]
    df['Prediction_Probability'] = predictions
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(file_path))
    df.to_excel(output_path, index=False)
    return output_path

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        processed_file_path = process_file(temp_path)
        download_filename = os.path.basename(processed_file_path)
        download_url = url_for('download_file', filename=download_filename)
        return jsonify({'downloadUrl': download_url})
    else:
        return jsonify({'error': 'Invalid file or file processing failed'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    response = query_openai_gpt(user_query)
    return jsonify({'response': response})

def query_openai_gpt(query):
    openai.api_key = app.config['OPENAI_API_KEY']
    try:
        completion = openai.Completion.create(
          engine="gpt-3.5-turbo",
          prompt=query,
          temperature=0.7,
          max_tokens=150
        )
        return completion.choices[0].text.strip()
    except Exception as e:
        return str(e)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls']

if __name__ == '__main__':
    app.run(debug=True)
