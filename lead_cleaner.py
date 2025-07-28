import json
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging

class LeadCleaner:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4.1-2025-04-14",
            openai_api_key=api_key,
            temperature=0.2
        )
        
    def clean_batch(self, batch_records: List[Dict]) -> List[Dict]:
        """Clean a batch of lead records and add validation flags - DEPRECATED: Use clean_csv_batch instead"""
        logging.warning("clean_batch method is deprecated, use clean_csv_batch for CSV processing")
        
        # Convert records back to CSV format for processing
        if not batch_records:
            return []
        
        # Get headers from first record
        headers = list(batch_records[0].keys())
        
        # Create CSV data
        csv_lines = [','.join(headers)]
        for record in batch_records:
            row = []
            for header in headers:
                value = str(record.get(header, ''))
                # Quote values that contain commas
                if ',' in value or '"' in value:
                    value = f'"{value.replace(chr(34), chr(34)+chr(34))}"'
                row.append(value)
            csv_lines.append(','.join(row))
        
        csv_data = '\n'.join(csv_lines)
        
        # Use the new CSV processing method
        return self.clean_csv_batch(csv_data)
    
    def process_json(self, json_file_path: str, output_path: str, batch_size: int = 1) -> str:
        """Process entire JSON file in batches"""
        return self.process_json_with_progress(json_file_path, output_path, batch_size, None)
    
    def process_json_with_progress(self, json_file_path: str, output_path: str, batch_size: int = 1, progress_callback=None) -> str:
        """Process entire JSON file in batches with progress reporting"""
        
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of records")
        
        total_records = len(data)
        total_batches = (total_records - 1) // batch_size + 1
        
        logging.info(f"Processing {total_records} records in {total_batches} batches of {batch_size}")
        
        cleaned_records = []
        leads_processed = 0
        
        # Process in batches
        for i in range(0, total_records, batch_size):
            batch_num = i // batch_size + 1
            batch = data[i:i+batch_size]
            
            logging.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Report progress before processing batch
            if progress_callback:
                progress_callback(batch_num, total_batches, leads_processed, total_records, batch_size)
            
            cleaned_batch = self.clean_batch(batch)
            cleaned_records.extend(cleaned_batch)
            
            leads_processed += len(batch)
            
            # Report progress after processing batch
            if progress_callback:
                progress_callback(batch_num, total_batches, leads_processed, total_records, batch_size)
        
        # Save to output file as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Cleaning complete. Output saved to {output_path}")
        return output_path 

    def process_csv_with_progress(self, csv_file_path: str, output_path: str, batch_size: int = 1, progress_callback=None) -> str:
        """Process CSV file directly, passing raw CSV data to LLM for processing"""
        
        # Read the CSV file as text
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            csv_content = f.read().strip()
        
        lines = csv_content.split('\n')
        if len(lines) < 2:
            raise ValueError("CSV file must contain header and at least one data row")
        
        header_line = lines[0]
        data_lines = lines[1:]
        
        # Filter out empty lines
        data_lines = [line for line in data_lines if line.strip()]
        
        total_records = len(data_lines)
        total_batches = (total_records - 1) // batch_size + 1
        
        logging.info(f"Processing {total_records} CSV records in {total_batches} batches of {batch_size}")
        
        cleaned_records = []
        leads_processed = 0
        
        # Process in batches
        for i in range(0, total_records, batch_size):
            batch_num = i // batch_size + 1
            batch_data_lines = data_lines[i:i+batch_size]
            
            # Create CSV batch with header + data rows
            csv_batch = header_line + '\n' + '\n'.join(batch_data_lines)
            
            logging.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Report progress before processing batch
            if progress_callback:
                progress_callback(batch_num, total_batches, leads_processed, total_records, batch_size)
            
            # Process the CSV batch directly
            cleaned_batch = self.clean_csv_batch(csv_batch)
            cleaned_records.extend(cleaned_batch)
            
            leads_processed += len(batch_data_lines)
            
            # Report progress after processing batch
            if progress_callback:
                progress_callback(batch_num, total_batches, leads_processed, total_records, batch_size)
        
        # Save to output file as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
        
        logging.info(f"CSV cleaning complete. Output saved to {output_path}")
        return output_path
    
    def clean_csv_batch(self, csv_data: str) -> List[Dict]:
        """Clean a batch of CSV data and return JSON records with validation flags"""
        
        system_prompt = """You are a data cleaning agent specializing in physiotherapy/healthcare lead data. Your task is to review CSV lead data and ensure each field contains the correct type of information.

EXPECTED CSV DATA FORMAT:
The data contains physiotherapy practice leads with columns like:
- Company: Business name (e.g., "PhysioWell AG", "Rehabilitation Center")
- Lead_Description: Business description or services offered
- Contact Person/FirstName/LastName/Salutation: Individual contact names
- Phone/Email/Website: Primary contact information
- Address/City/PostalCode/Country: Location information  
- Physio_Tags: Treatment specializations (e.g., "Sports Therapy, Manual Therapy")
- Physio_Specialization: Specific techniques offered
- Lead_language: Language for communication
- Status: Lead qualification status
- Multiple optional fields: Phone_2, Email_2, Website_2, Profile URL_2, Phone_3, etc.

CRITICAL CLEANING RULES:

1. **PHONE NUMBERS**: Look for patterns like +41 79 123 45 67, 555-123-4567, +49 176 888 99 11
   - Primary phone → "phone" field
   - Additional phones → "phone_2", "phone_3", etc. (if data for these fields exist)
   - Move any phone numbers that appear in wrong fields (like in company name) to the appropriate field

2. **EMAIL ADDRESSES**: Look for @domain.com patterns
   - Primary email → "email" field  
   - Additional emails → "email_2", "email_3", etc. (if these fields exist)
   - Clean emails from wrong fields and move them to the appropriate field

3. **WEBSITES**: Look for www.domain.com, https://domain.com, domain.com patterns
   - Main website → "website" field
   - Additional websites → "website_2", "website_3", etc. (if these fields exist)

4. **NAMES**: 
   - Full names like "Lisa Meier" → split appropriately into firstname/lastname 
   - Contact person names → "contact person" field
   - Company names → "company" field (NOT in name fields)

5. **ADDRESSES**: Clean and organize address components
   - Full addresses → "address" field
   - City names → "city" field  
   - Postal codes → "postalcode" field
   - Countries → "country" field

6. **MISSING VALUES**: 
   - Fill out field with "NA" if data is genuinely not available, BUT pay close attention to all data, there might be data hidden in different fields. -> Maybe the contact persons name is in the Adress.
   - DO NOT invent or create data
   - DO NOT move data unless you're confident it belongs in another field

7. **WRONG ENTRIES**: 
   - If an email appears in the phone field, move it to appropriate email field
   - If a phone appears in the email field, move it to appropriate phone field  
   - If a website appears in wrong field, move it to appropriate website field
   - If a company name appears in a person name field, move it to company field

9. **MULTIPLE CONTACT FIELDS & DUPLICATE REMOVAL**: 
   - Use Phone_2, Email_2, Website_2, Profile URL_2, Phone_3, etc. ONLY for genuinely different contact info
   - These fields are completely optional - only use if data exists AND is different from primary fields
   - Don't create empty numbered fields
   - **CRITICAL: NEVER DUPLICATE VALUES** - if Phone_2 has the same value as Phone, DELETE Phone_2 completely
   - **REMOVE ALL DUPLICATES**: If any numbered field (Phone_2, Email_2, Website_2, etc.) contains the same value as the primary field, remove the duplicate entirely
   - Set duplicate fields to "NA" rather than keeping duplicate values
   - Example: if Phone="123456" and Phone_2="123456", then set Phone_2="NA"

10. **PRESERVE EXISTING DATA**: 
   - If data is already in the correct field, leave it unchanged unless there is a 1 to 1 duplicate in the additional fields.
   - Only move data when there's a clear mismatch
   - Better to leave questionable data in place than to move it incorrectly

11. **SPECIAL FIELDS**:
    - Keep physiotherapy-specific content in Physio_Tags, Physio_Specialization
    - Preserve language information in Lead_language
    - Maintain business descriptions in Lead_Description

VALIDATION FLAGS:
- 0: No changes needed - data is already in correct fields
- 1: Made corrections - moved data between fields or cleaned formatting
- 2: Suspicious/unclear - data quality issues, multiple problems, or unclear field placement. Use this if you are not sure about the data.

IMPORTANT: You will receive CSV data (header row + data rows). Return ONLY a valid JSON array where each object represents one cleaned record with a "validation_flag" field added. No explanations, no markdown, no extra text."""

        human_prompt = f"""Clean this CSV lead data and return ONLY the JSON array:

{csv_data}

Expected output format: [{{ "field1": "value1", "field2": "value2", "validation_flag": 0 }}, ...]"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_content = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]  # Remove ```json
            elif response_content.startswith('```'):
                response_content = response_content[3:]   # Remove ```
            
            if response_content.endswith('```'):
                response_content = response_content[:-3]  # Remove closing ```
            
            response_content = response_content.strip()
            
            # Try to find JSON array in the response
            if '[' in response_content and ']' in response_content:
                start_idx = response_content.find('[')
                end_idx = response_content.rfind(']') + 1
                json_str = response_content[start_idx:end_idx]
            else:
                json_str = response_content
            
            # Clean up any remaining formatting issues
            json_str = json_str.replace('\n', '').replace('\r', '')
            
            cleaned_data = json.loads(json_str)
            
            # Validate that we got a list
            if not isinstance(cleaned_data, list):
                raise ValueError("Response is not a list")
            
            # Ensure validation_flag is string and clean data
            for record in cleaned_data:
                if isinstance(record, dict):
                    record['validation_flag'] = str(record.get('validation_flag', '2'))
                    # Clean all values to be strings
                    for key, value in record.items():
                        if key != 'validation_flag':
                            record[key] = str(value) if value is not None else ""
            
            return cleaned_data
            
        except Exception as e:
            logging.error(f"Error processing CSV batch: {e}")
            if 'response' in locals():
                logging.error(f"Raw response: {response.content[:500]}...")  # Log first 500 chars
            
            # Fallback: return original data with flag 2 (needs review)
            # Parse CSV data as fallback
            lines = csv_data.strip().split('\n')
            if len(lines) >= 2:
                import csv as csv_module
                from io import StringIO
                
                reader = csv_module.reader(StringIO(csv_data), quotechar='"')
                headers = next(reader)
                
                fallback_data = []
                for row in reader:
                    fallback_record = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            fallback_record[header.lower().strip()] = str(row[i]) if row[i] else ""
                        else:
                            fallback_record[header.lower().strip()] = ""
                    fallback_record['validation_flag'] = '2'  # Needs review
                    fallback_data.append(fallback_record)
                
                logging.info(f"Fallback applied to {len(fallback_data)} records")
                return fallback_data
            
            return [] 