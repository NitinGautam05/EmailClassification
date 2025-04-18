# Email Classification System for Support Team

This project implements a support email classification system with PII masking capabilities. It processes incoming support emails, detects and masks personally identifiable information (PII), classifies emails into predefined categories, and exposes the functionality through a REST API.

## Features

- **PII/PCI Detection and Masking**: Detects and masks sensitive information including:
  - Full names
  - Email addresses
  - Phone numbers
  - Date of birth
  - Aadhar card numbers
  - Credit/debit card numbers
  - CVV numbers
  - Card expiry dates

- **Email Classification**: Categorizes emails into:
  - Incident
  - Request
  - Change
  - Problem

- **REST API**: Provides an endpoint for classifying emails and masking PII.

## Project Structure

- `app.py`: Main application entry point
- `api.py`: FastAPI implementation
- `models.py`: Classification model implementation
- `utils.py`: Utility functions for PII/PCI masking
- `requirements.txt`: Project dependencies

## Setup and Installation

### Prerequisites

- Python 3.8+

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/NitinGautam05/email-classification.git
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the required spaCy model:
   ```
   python -m spacy download en_core_web_md
   ```

5. Run the application:
   ```
   python app.py
   ```

   The API will be available at http://localhost:8000/docs

### Deployment on Hugging Face Spaces

This project is designed to be easily deployed on Hugging Face Spaces:

1. Fork this repository to your GitHub account.
2. Create a new Space on Hugging Face.
3. Connect your GitHub repository to the Space.
4. Select "Streamlit" as the Space SDK.
5. Configure the Space to install all dependencies from `requirements.txt`.

## API Usage

### Classify Email Endpoint

**Endpoint**: `POST /classify_email`

**Request Body**:
```json
{
  "email_body": "Hello, my name is John Doe, and my email is johndoe@example.com. I'm having issues with my billing statement."
}
```

**Response**:
```json
{
  "input_email_body": "Hello, my name is John Doe, and my email is johndoe@example.com. I'm having issues with my billing statement.",
  "list_of_masked_entities": [
    {
      "position": [17, 25],
      "classification": "full_name",
      "entity": "John Doe"
    },
    {
      "position": [41, 60],
      "classification": "email",
      "entity": "johndoe@example.com"
    }
  ],
  "masked_email": "Hello, my name is [full_name], and my email is [email]. I'm having issues with my billing statement.",
  "category_of_the_email": "Problem"
}
```

## Implementation Details

### PII/PCI Detection

The system uses a combination of:
- Regular expressions for structured data like email addresses and card numbers
- spaCy's named entity recognition for detecting names
- Custom pattern matching for specific formats

### Email Classification

The classification model uses:
- TF-IDF vectorization for feature extraction
- Random Forest classifier for categorization
- NLTK for text preprocessing

## License

[MIT License](LICENSE)
