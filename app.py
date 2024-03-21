from flask import Flask, request, send_file, render_template, jsonify, url_for
import pandas as pd
import pickle
import io
import os
from werkzeug.utils import secure_filename
import openai

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'temp'
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a random secret key

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your model
model_path = 'models/logistic_regression_model.pkl'
loaded_model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

def handle_missing_values(data):
    data['HIGH EQUITY'] = data['HIGH EQUITY'].fillna(0)
    data['INTER FAMILY TRANSFER']=data['INTER FAMILY TRANSFER'].fillna(0)
    data['55+']=data['55+'].fillna(0)
    data['TAXES']=data['TAXES'].fillna(0)
    data=data.fillna(0)
    return data

def process_file(file_path):
    # Process the uploaded file
    df = pd.read_excel(file_path)
    tag_columns = ['FOLIO', 'ABSENTEE', 'HIGH EQUITY', 'DOWNSIZING', 'VACANT', '55+', 'INTER FAMILY TRANSFER', 'TAXES']
    df = df[tag_columns]
    df = df.drop_duplicates('FOLIO')
    df = handle_missing_values(df)
    data = df.drop('FOLIO', axis=1)
    predictions = loaded_model.predict_proba(data)[:, 1]
    df['Prediction_Probability'] = predictions
    
    # Convert DataFrame to CSV and save for download
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(file_path))
    df = df.sort_values('Prediction_Probability',ascending= False)
    df.to_excel(output_path, index=False)
    
    return output_path

def generate_insights_from_data(data,prompt_text):
    """Generates insights from the DataFrame using OpenAI's GPT."""
    openai.api_key = os.getenv('API_KEY')
    print(openai.api_key)
    summary = data.describe().to_string()
    try:
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",  # You may want to use the latest model here
            prompt=prompt_text+ f'Here is the data : \n {data}',
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating insights: {e}")
        return None
    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and loaded_model:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        processed_file_path = process_file(temp_path)
        if processed_file_path:
            download_filename = os.path.basename(processed_file_path)
            download_url = url_for('download_file', filename=download_filename)
            return jsonify({'url': download_url})
        else:
            return jsonify({'error': 'Processing failed'}), 500
    else:
        return jsonify({'error': 'Model not loaded or file processing issue'}), 500

@app.route('/generate_insights', methods=['POST'])
def generate_insights():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try : 
    # Process the file to DataFrame
        data = pd.read_excel(file)
    except: 
        data = pd.read_csv(file)
        
    insights = generate_insights_from_data(data, 'Please generate the insights from the data within 50 words. Bring more stats into explanation. Explain variables. Keep it strictly under 50 words')
    
    if insights:
        # Return the generated insights
        return render_template('insights.html', insights=insights)
    else:
        return jsonify({'error': 'Failed to generate insights'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    directory = app.config['UPLOAD_FOLDER']
    try:
        return send_file(os.path.join(directory, filename), as_attachment=True)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=False)
