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
        
        # Read the processed JSON data
        with open(output_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        if not isinstance(processed_data, list):
            return jsonify({'error': 'Invalid results file format'}), 500
        
        # Get summary statistics
        total_records = len(processed_data)
        unchanged = len([r for r in processed_data if r.get('validation_flag') == '0'])
        corrected = len([r for r in processed_data if r.get('validation_flag') == '1'])
        suspicious = len([r for r in processed_data if r.get('validation_flag') == '2'])
        
        # Clean records for JSON serialization
        cleaned_records = []
        for record in processed_data:
            cleaned_record = {}
            for key, value in record.items():
                # Ensure all values are clean strings
                if value is None or value == 'nan':
                    cleaned_record[key] = ''
                else:
                    # Remove any problematic characters
                    cleaned_record[key] = str(value).replace('\x00', '').strip()
            cleaned_records.append(cleaned_record)
        
        result = {
            'success': True,
            'output_filename': filename,
            'summary': {
                'total': total_records,
                'unchanged': unchanged,
                'corrected': corrected,
                'suspicious': suspicious
            },
            'records': cleaned_records
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
    
    file_ext = file.filename.lower().split('.')[-1]
    if file_ext not in ['csv', 'json']:
        return jsonify({'error': 'Only CSV (with any delimiter: , ; | tab) and JSON files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # For CSV files, keep them as CSV for LLM processing
        if file_ext == 'csv':
            # Just validate the CSV can be read and get record count
            total_records = validate_csv_file(upload_path)
            processing_filename = filename  # Keep original CSV filename
        else:
            # For JSON files, validate and potentially clean
            json_filename = filename.rsplit('.', 1)[0] + '.json'
            json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            total_records = validate_and_clean_json(upload_path, json_path)
            processing_filename = json_filename
        
        # Reset processing status
        processing_status.update({
            'is_processing': False,
            'current_batch': 0,
            'total_batches': 0,
            'leads_processed': 0,
            'total_leads': total_records,
            'filename': processing_filename,
            'status_message': f'File uploaded: {total_records} leads ready for processing',
            'completed': False,
            'error': None
        })
        
        result = {
            'success': True,
            'filename': processing_filename,
            'total_records': total_records,
            'message': f'File uploaded successfully. {total_records} leads ready for processing.'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

def validate_csv_file(csv_path):
    """Validate CSV file and return record count without converting to JSON"""
    try:
        # Detect delimiter for validation
        delimiter = detect_csv_delimiter(csv_path)
        logging.info(f"CSV validation using delimiter: '{delimiter}'")
        
        # Try reading with detected delimiter to validate and count records
        import csv
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar='"')
            headers = next(reader)  # Read headers
            
            # Count data rows
            record_count = 0
            for row in reader:
                if any(field.strip() for field in row):  # Skip completely empty rows
                    record_count += 1
        
        logging.info(f"CSV validation successful: {len(headers)} columns, {record_count} records")
        return record_count
        
    except Exception as e:
        logging.error(f"CSV validation failed: {e}")
        raise Exception(f"Invalid CSV file: {str(e)}")

def detect_csv_delimiter(csv_path):
    """Detect the CSV delimiter by analyzing the first few lines with strong validation"""
    import csv
    
    # Common delimiters to test, prioritized by likelihood
    delimiters = [',', ';', '\t', '|']  # Put pipe last as it's least likely for CSV
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Read more sample data for better detection
            sample = f.read(2048)
            
        # Use csv.Sniffer to detect delimiter first
        sniffer = csv.Sniffer()
        try:
            # Prioritize comma and semicolon in sniffer
            dialect = sniffer.sniff(sample, delimiters=',;\t|')
            detected_delimiter = dialect.delimiter
            
            # STRONG validation of sniffer result
            lines = sample.split('\n')
            if len(lines) >= 2:
                # Use csv.reader to properly count fields (handles quotes)
                from io import StringIO
                
                # Test with detected delimiter using csv.reader
                test_sample = '\n'.join(lines[:2])  # Just header and first data line
                test_reader = csv.reader(StringIO(test_sample), delimiter=detected_delimiter, quotechar='"')
                
                try:
                    headers = next(test_reader)
                    first_row = next(test_reader)
                    
                    header_count = len(headers)
                    data_count = len(first_row)
                    
                    # Strong validation criteria
                    if (header_count > 1 and data_count > 1 and 
                        abs(header_count - data_count) <= 1 and
                        header_count >= 3):  # Must have at least 3 fields for lead data
                        
                        logging.info(f"CSV Sniffer validated: '{detected_delimiter}' (headers: {header_count}, data: {data_count})")
                        return detected_delimiter
                    else:
                        logging.warning(f"CSV Sniffer validation failed: '{detected_delimiter}' (headers: {header_count}, data: {data_count})")
                        
                except Exception as reader_error:
                    logging.warning(f"CSV reader validation failed for '{detected_delimiter}': {reader_error}")
            
        except Exception as sniffer_error:
            logging.warning(f"CSV Sniffer failed: {sniffer_error}")
        
        # Enhanced manual fallback with stronger validation
        lines = sample.split('\n')
        if len(lines) < 2:
            logging.warning("Not enough lines for delimiter detection, defaulting to comma")
            return ','
        
        logging.info("Using enhanced manual delimiter detection")
        
        # Test each delimiter with stronger criteria
        best_delimiter = ','
        best_score = 0
        delimiter_results = {}
        
        for delimiter in delimiters:
            try:
                # Use csv.reader for accurate field counting
                from io import StringIO
                test_sample = '\n'.join(lines[:3])  # First 3 lines
                test_reader = csv.reader(StringIO(test_sample), delimiter=delimiter, quotechar='"')
                
                field_counts = []
                line_count = 0
                
                for row in test_reader:
                    if row:  # Skip empty rows
                        field_counts.append(len(row))
                        line_count += 1
                        
                if field_counts and line_count >= 2:
                    avg_fields = sum(field_counts) / len(field_counts)
                    variance = sum((x - avg_fields) ** 2 for x in field_counts) / len(field_counts)
                    
                    # Enhanced scoring with strong bias against problematic delimiters
                    if avg_fields >= 3 and avg_fields < 100:  # Must have at least 3 fields
                        # Heavily penalize single-field results (likely wrong delimiter)
                        if avg_fields < 2:
                            score = 0
                        else:
                            # Base score calculation
                            consistency_bonus = 1 / (1 + variance)
                            field_count_bonus = min(avg_fields / 10, 1)  # Optimal around 10 fields
                            
                            # Special bonuses and penalties
                            delimiter_bonus = 1.0
                            if delimiter == ',':
                                delimiter_bonus = 1.5  # Strong preference for comma
                            elif delimiter == ';':
                                delimiter_bonus = 1.2  # Some preference for semicolon
                            elif delimiter == '|':
                                delimiter_bonus = 0.5  # Penalty for pipe (rarely used in CSV)
                            
                            score = avg_fields * consistency_bonus * field_count_bonus * delimiter_bonus
                        
                        delimiter_results[delimiter] = {
                            'avg_fields': avg_fields,
                            'variance': variance,
                            'score': score,
                            'field_counts': field_counts
                        }
                        
                        logging.info(f"Delimiter '{delimiter}': avg_fields={avg_fields:.1f}, variance={variance:.2f}, score={score:.2f}")
                        
                        if score > best_score:
                            best_score = score
                            best_delimiter = delimiter
                            
            except Exception as e:
                logging.warning(f"Error testing delimiter '{delimiter}': {e}")
                continue
        
        # Final validation of chosen delimiter
        if best_delimiter and best_score > 0:
            result = delimiter_results.get(best_delimiter, {})
            avg_fields = result.get('avg_fields', 0)
            
            # Absolutely prevent single-field results
            if avg_fields < 2:
                logging.error(f"Chosen delimiter '{best_delimiter}' produces too few fields ({avg_fields}), forcing comma")
                best_delimiter = ','
            
            # Extra validation: test if delimiter actually works
            try:
                from io import StringIO
                test_sample = '\n'.join(lines[:2])
                test_reader = csv.reader(StringIO(test_sample), delimiter=best_delimiter, quotechar='"')
                headers = next(test_reader)
                data = next(test_reader)
                
                if len(headers) >= 3 and len(data) >= 3:
                    logging.info(f"Final validation passed: delimiter '{best_delimiter}' produces {len(headers)} headers, {len(data)} data fields")
                    return best_delimiter
                else:
                    logging.error(f"Final validation failed: delimiter '{best_delimiter}' produces {len(headers)} headers, {len(data)} data fields")
                    
            except Exception as validation_error:
                logging.error(f"Final validation error for '{best_delimiter}': {validation_error}")
        
        # Ultimate fallback with explicit check
        logging.warning("All delimiter detection methods failed or produced invalid results")
        
        # Try comma as last resort
        try:
            from io import StringIO
            test_sample = '\n'.join(lines[:2])
            test_reader = csv.reader(StringIO(test_sample), delimiter=',', quotechar='"')
            headers = next(test_reader)
            data = next(test_reader)
            
            if len(headers) >= 2 and len(data) >= 2:
                logging.info(f"Fallback to comma successful: {len(headers)} headers, {len(data)} data fields")
                return ','
                
        except Exception:
            pass
        
        # Absolute last resort
        logging.error("Even comma fallback failed, but returning comma anyway")
        return ','
        
    except Exception as e:
        logging.error(f"Critical error in delimiter detection: {e}")
        return ','  # Always return comma as final fallback


# Note: CSV to JSON conversion functions removed - we now pass CSV data directly to LLM
# This eliminates delimiter detection issues and makes the system more robust


def validate_and_clean_json(json_path, output_json_path):
    """Validate and clean JSON input"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise Exception("JSON must contain an array of records")
        
        # Clean the records
        cleaned_records = []
        for record in data:
            if isinstance(record, dict):
                # Clean each field
                cleaned_record = {}
                for key, value in record.items():
                    # Ensure keys are strings and clean values
                    clean_key = str(key).lower().strip()
                    clean_value = str(value) if value is not None else ""
                    cleaned_record[clean_key] = clean_value
                cleaned_records.append(cleaned_record)
        
        # Save cleaned JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
        
        return len(cleaned_records)
        
    except Exception as e:
        logging.error(f"Error validating JSON: {e}")
        raise Exception(f"Invalid JSON format: {str(e)}")

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
            # Determine input file type and set output filename
            file_ext = upload_path.lower().split('.')[-1]
            if file_ext == 'csv':
                output_filename = f"cleaned_{filename.rsplit('.', 1)[0]}.json"
            else:
                output_filename = f"cleaned_{filename.rsplit('.', 1)[0]}.json"
            
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
            if file_ext == 'csv':
                cleaner.process_csv_with_progress(upload_path, output_path, progress_callback=update_progress)
            else:
                cleaner.process_json_with_progress(upload_path, output_path, progress_callback=update_progress)
            
            # Read the processed data for final results
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            # Get summary statistics
            total_records = len(processed_data)
            unchanged = len([r for r in processed_data if r.get('validation_flag') == '0'])
            corrected = len([r for r in processed_data if r.get('validation_flag') == '1'])
            suspicious = len([r for r in processed_data if r.get('validation_flag') == '2'])
            
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

def update_progress(current_batch, total_batches, leads_processed, total_leads, batch_size=1):
    """Callback function to update processing progress"""
    global processing_status
    
    processing_status.update({
        'current_batch': current_batch,
        'total_batches': total_batches,
        'leads_processed': leads_processed,
        'status_message': f'Processing record {current_batch}/{total_batches} - {leads_processed}/{total_leads} leads cleaned'
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
        
        # Read the current JSON file
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # Update the specific record
        if 0 <= record_index < len(records):
            for key, value in updated_record.items():
                records[record_index][key] = value
            
            # Save the updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid record index'}), 400
        
    except Exception as e:
        logging.error(f"Error updating record: {e}")
        return jsonify({'error': f'Error updating record: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 