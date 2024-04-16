from flask import Flask, request, send_file, render_template, jsonify, url_for,session
import pandas as pd
import pickle
import io
import os
from werkzeug.utils import secure_filename
import openai
from tabulate import tabulate
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
    data['INTER FAMILY TRANSFER'] = data['INTER FAMILY TRANSFER'].fillna(0)
    data['55+'] = data['55+'].fillna(0)
    data['TAXES'] = data['TAXES'].fillna(0)
    data = data.fillna(0)
    return data

def process_file(file_path):
    # Determine file type from extension to handle both CSV and Excel files
    file_extension = file_path.split('.')[-1].lower()
    if file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    elif file_extension == 'csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format for processing.")
    
    # Process the DataFrame
    tag_columns = ['FOLIO', 'ABSENTEE', 'HIGH EQUITY', 'DOWNSIZING', 'VACANT', '55+', 'INTER FAMILY TRANSFER', 'TAXES']
    df = df[tag_columns]
    df = df.drop_duplicates('FOLIO')
    df = handle_missing_values(df)
    data = df.drop('FOLIO', axis=1)
    predictions = loaded_model.predict_proba(data)[:, 1]
    df['Prediction_Probability'] = predictions
    df = df.sort_values('Prediction_Probability',ascending= False)
    
    # Save the processed file
    output_extension = 'xlsx' if file_extension in ['xls', 'xlsx'] else 'csv'
    output_filename = f'processed_{os.path.basename(file_path).split(".")[0]}.{output_extension}'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    if output_extension == 'csv':
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)
    
    return output_path

def generate_insights_from_data(data, prompt_text):
    """Generates insights from the DataFrame using OpenAI's GPT."""
    openai.api_key = os.getenv('API_KEY')
    if len(data)>1000:
        summary = tabulate(data.describe(), headers='keys', tablefmt='pipe', showindex=True)
    elif len==0:
        summary = " No data is present. Just answer the"
    else:
        summary = tabulate(data, headers='keys', tablefmt='pipe', showindex=True)
    try:
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt_text + f' Here is the data: \n{summary}',
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
    
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    try:
        processed_file_path = process_file(temp_path)
        download_filename = os.path.basename(processed_file_path)
        download_url = url_for('download_file', filename=download_filename)
        return jsonify({'url': download_url})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/generate_insights', methods=['POST'])
def generate_insights():
    prompt = request.form.get('prompt')
    # Check if there is a file path stored in the session
    if 'file_path' in session:
        temp_path = session['file_path']
        print(temp_path)
        file_extension = temp_path.split('.')[-1].lower()

        if file_extension in ['xls', 'xlsx']:
            data = pd.read_excel(temp_path)
        elif file_extension == 'csv':
            data = pd.read_csv(temp_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    else:
        try:            
            if file_extension in ['xls', 'xlsx']:
                data = pd.read_excel(session['file_path'])
            elif file_extension == 'csv':
                data = pd.read_csv(temp_path)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400 # Create an empty DataFrame or load a default
        except:
            data = pd.DataFrame()
    insights = generate_insights_from_data(data, prompt)
    if insights:
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
