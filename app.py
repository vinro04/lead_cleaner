from flask import Flask, request, render_template, jsonify, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from lead_cleaner import LeadCleaner
import tempfile
import json
import threading
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the lead cleaner
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

cleaner = LeadCleaner(openai_api_key)

# Progress tracking
processing_status = {
    'is_processing': False,
    'current_batch': 0,
    'total_batches': 0,
    'leads_processed': 0,
    'total_leads': 0,
    'filename': '',
    'status_message': '',
    'completed': False,
    'error': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    return jsonify(processing_status)

@app.route('/results/<filename>')
def get_results(filename):
    try:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(output_path):
            return jsonify({'error': 'Results file not found'}), 404
        
        # Read the processed data
        processed_df = pd.read_csv(output_path)
        
        # Clean the data for JSON serialization
        # Replace NaN values with empty strings
        processed_df = processed_df.fillna('')
        
        # Convert all data to strings to avoid serialization issues
        for column in processed_df.columns:
            processed_df[column] = processed_df[column].astype(str)
        
        # Get summary statistics
        total_records = len(processed_df)
        unchanged = len(processed_df[processed_df['validation_flag'] == '0'])
        corrected = len(processed_df[processed_df['validation_flag'] == '1'])
        suspicious = len(processed_df[processed_df['validation_flag'] == '2'])
        
        # Convert to records with additional sanitization
        records = []
        for _, row in processed_df.iterrows():
            record = {}
            for col, val in row.items():
                # Ensure all values are serializable strings
                if pd.isna(val) or val == 'nan':
                    record[col] = ''
                else:
                    # Remove any problematic characters
                    record[col] = str(val).replace('\x00', '').strip()
            records.append(record)
        
        result = {
            'success': True,
            'output_filename': filename,
            'summary': {
                'total': total_records,
                'unchanged': unchanged,
                'corrected': corrected,
                'suspicious': suspicious
            },
            'records': records
        }
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error getting results: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Error loading results: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Read file to get total record count
        df = pd.read_csv(upload_path)
        total_records = len(df)
        
        # Reset processing status
        processing_status.update({
            'is_processing': False,
            'current_batch': 0,
            'total_batches': 0,
            'leads_processed': 0,
            'total_leads': total_records,
            'filename': filename,
            'status_message': f'File uploaded: {total_records} leads ready for processing',
            'completed': False,
            'error': None
        })
        
        result = {
            'success': True,
            'filename': filename,
            'total_records': total_records,
            'message': 'File uploaded successfully. Click "Run Now" to start processing.'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_file():
    global processing_status
    
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress'}), 400
    
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(upload_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Start processing in background thread
    def process_in_background():
        global processing_status
        
        try:
            output_filename = f"cleaned_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            processing_status.update({
                'is_processing': True,
                'current_batch': 0,
                'leads_processed': 0,
                'status_message': 'Starting processing...',
                'completed': False,
                'error': None
            })
            
            # Process the file with progress callback
            cleaner.process_csv_with_progress(upload_path, output_path, progress_callback=update_progress)
            
            # Read the processed data for final results
            processed_df = pd.read_csv(output_path)
            
            # Get summary statistics
            total_records = len(processed_df)
            unchanged = len(processed_df[processed_df['validation_flag'] == 0])
            corrected = len(processed_df[processed_df['validation_flag'] == 1])
            suspicious = len(processed_df[processed_df['validation_flag'] == 2])
            
            processing_status.update({
                'is_processing': False,
                'completed': True,
                'status_message': f'Processing complete! {total_records} leads processed.',
                'output_filename': output_filename,
                'total_records': total_records,
                'unchanged': unchanged,
                'corrected': corrected,
                'suspicious': suspicious
            })
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            processing_status.update({
                'is_processing': False,
                'completed': False,
                'error': str(e),
                'status_message': f'Error: {str(e)}',
                'current_batch': 0,
                'total_batches': 0,
                'leads_processed': 0
            })
    
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Processing started'})

def update_progress(current_batch, total_batches, leads_processed, total_leads, batch_size=10):
    """Callback function to update processing progress"""
    global processing_status
    
    processing_status.update({
        'current_batch': current_batch,
        'total_batches': total_batches,
        'leads_processed': leads_processed,
        'status_message': f'Processing batch {current_batch}/{total_batches} - {leads_processed}/{total_leads} leads cleaned'
    })

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

@app.route('/update_record', methods=['POST'])
def update_record():
    try:
        data = request.get_json()
        filename = data.get('filename')
        record_index = data.get('index')
        updated_record = data.get('record')
        
        # Read the current file
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        df = pd.read_csv(file_path)
        
        # Update the specific record
        for key, value in updated_record.items():
            df.at[record_index, key] = value
        
        # Save the updated file
        df.to_csv(file_path, index=False)
        
        return jsonify({'success': True})
        
    except Exception as e:
        logging.error(f"Error updating record: {e}")
        return jsonify({'error': f'Error updating record: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 