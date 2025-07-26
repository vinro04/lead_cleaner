# Lead Data Cleaner

A simple web application that uses LangChain and GPT-4o Mini to clean and validate lead data from CSV files.

## Features

- Drag & drop CSV file upload
- Batch processing (10 leads at a time) using GPT-4o Mini
- Intelligent data cleaning and validation
- Color-coded results display with validation flags:
  - Green (0): No changes needed
  - Yellow (1): Corrections made
  - Red (2): Needs manual review
- Edit records directly in the interface
- Download cleaned CSV file

## Setup for Replit

1. Fork this repository to Replit
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Run the application using `main.py`
4. The web interface will be available at your Replit URL

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with your OpenAI API key

3. Run the application:
   ```bash
   python app.py
   ```

4. Open http://localhost:5000 in your browser

## Usage

1. Drop or select a CSV file with lead data
2. Wait for processing (files are processed in batches of 10)
3. Review the results with color-coded validation flags
4. Edit any records that need manual correction
5. Download the cleaned CSV file

The system will automatically detect and correct common data placement issues like emails in phone columns, phone numbers in name fields, etc. 