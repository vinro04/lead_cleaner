import pandas as pd
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import logging

class LeadCleaner:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0
        )
        
    def clean_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Clean a batch of leads and add validation flags"""
        
        # Convert batch to JSON for LLM processing
        batch_records = batch_df.to_dict('records')
        
        system_prompt = """You are a data cleaning agent. Your task is to review lead data and ensure each field contains the correct type of information.

EXAMPLE OF CLEAN DATA FORMAT:
Lead Status,Name,Email,Phone,Website,Company,Language,Email Language,Email Salutation,Physio Tags
New,Lisa Meier,lisa.meier@physiowell.ch,+41 79 123 45 67,www.physiowell.ch,PhysioWell AG,German,German,Liebe Frau Meier,"Sportphysio, Manuelle Therapie"
Contacted,Mark Jensen,mark.j@recoverymove.de,+49 176 888 99 11,www.recoverymove.de,RecoveryMove GmbH,German,German,Lieber Herr Jensen,"Krankengymnastik, Kinesiotaping"

STRICT RULES:
1. Phone numbers should be in the "phone" column (look for numbers like +41 79 123 45 67, 555-123-4567, etc.)
2. Email addresses should be in the "email" column (look for @domain.com format)
3. Names should be in the "name" column (look for first/last names like "Lisa Meier")
4. Websites should be in the "website" column (look for www.domain.com or http formats)
5. Company names should be in the "company" column
6. Its possible that there are additional columns that are not in the example. Proceed with the columns that are present and dont get confused by them.
7. Only rearrange existing data - DO NOT create new data
8. You may leave fields empty ONLY if you are 100 percent sure the data is not available anywhere in the row
9. IMPORTANT: It's better to have a wrong entry than to delete data that should be in another field
10. If data looks correct, leave it unchanged

VALIDATION FLAGS:
- 0: No changes needed - data is in correct columns
- 1: Made corrections - swapped data between columns
- 2: Suspicious/unclear - data doesn't fit any column clearly or multiple issues

CRITICAL: Return ONLY a valid JSON array. No explanations, no markdown, no extra text."""

        human_prompt = f"""Clean this lead data and return ONLY the JSON array:

{json.dumps(batch_records, indent=2)}

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
            
            # Validate that we got a list and it has the right number of records
            if not isinstance(cleaned_data, list):
                raise ValueError("Response is not a list")
            
            if len(cleaned_data) != len(batch_records):
                logging.warning(f"Record count mismatch: expected {len(batch_records)}, got {len(cleaned_data)}")
                # Try to match as many as possible
                cleaned_data = cleaned_data[:len(batch_records)]
            
            # Convert back to DataFrame
            cleaned_df = pd.DataFrame(cleaned_data)
            
            # Ensure validation_flag is string and clean data
            cleaned_df['validation_flag'] = cleaned_df['validation_flag'].astype(str)
            cleaned_df = cleaned_df.fillna('')
            
            # Clean all other columns to be strings
            for column in cleaned_df.columns:
                if column != 'validation_flag':
                    cleaned_df[column] = cleaned_df[column].astype(str)
            
            return cleaned_df
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            if 'response' in locals():
                logging.error(f"Raw response: {response.content[:500]}...")  # Log first 500 chars
            
            # Fallback: return original data with flag 2 (needs review)
            fallback_df = batch_df.copy()
            fallback_df['validation_flag'] = '2'  # Use string for consistency
            
            # Clean the data to avoid JSON serialization issues
            fallback_df = fallback_df.fillna('')
            for column in fallback_df.columns:
                if column != 'validation_flag':
                    fallback_df[column] = fallback_df[column].astype(str)
            
            logging.info(f"Fallback applied to {len(fallback_df)} records")
            return fallback_df
    
    def process_csv(self, csv_file_path: str, output_path: str, batch_size: int = 10) -> str:
        """Process entire CSV file in batches"""
        return self.process_csv_with_progress(csv_file_path, output_path, batch_size, None)
    
    def process_csv_with_progress(self, csv_file_path: str, output_path: str, batch_size: int = 10, progress_callback=None) -> str:
        """Process entire CSV file in batches with progress reporting"""
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        total_records = len(df)
        total_batches = (total_records - 1) // batch_size + 1
        
        logging.info(f"Processing {total_records} records in {total_batches} batches of {batch_size}")
        
        cleaned_batches = []
        leads_processed = 0
        
        # Process in batches
        for i in range(0, total_records, batch_size):
            batch_num = i // batch_size + 1
            batch = df.iloc[i:i+batch_size].copy()
            
            logging.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Report progress before processing batch
            if progress_callback:
                progress_callback(batch_num, total_batches, leads_processed, total_records, batch_size)
            
            cleaned_batch = self.clean_batch(batch)
            cleaned_batches.append(cleaned_batch)
            
            leads_processed += len(batch)
            
            # Report progress after processing batch
            if progress_callback:
                progress_callback(batch_num, total_batches, leads_processed, total_records, batch_size)
        
        # Combine all cleaned batches
        final_df = pd.concat(cleaned_batches, ignore_index=True)
        
        # Save to output file
        final_df.to_csv(output_path, index=False)
        
        logging.info(f"Cleaning complete. Output saved to {output_path}")
        return output_path 