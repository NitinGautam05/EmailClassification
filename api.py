from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import re
import logging
from fastapi.middleware.cors import CORSMiddleware

from utils import PIIMasker, preprocess_email_text
from models import EmailClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailRequest(BaseModel):
    email_body: str

class Entity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[Entity]
    masked_email: str
    category_of_the_email: str

app = FastAPI(
    title="Email Classification API",
    description="API for classifying support emails and masking personally identifiable information",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize variables (will be loaded on first request)
pii_masker = None
classifier = None
model_path = "models/email_classifier.pkl"

def get_pii_masker():
    global pii_masker
    if pii_masker is None:
        logger.info("Loading PIIMasker...")
        pii_masker = PIIMasker()
    return pii_masker

def get_classifier():
    global classifier
    if classifier is None:
        logger.info("Loading EmailClassifier...")
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Creating a dummy classifier.")
            # Create a dummy classifier that always returns "unknown" category
            class DummyClassifier:
                def predict(self, text):
                    return "unknown"
            classifier = DummyClassifier()
        else:
            classifier = EmailClassifier(model_path)
    return classifier

@app.get("/")
async def root():
    """Root endpoint required by Hugging Face for health checks"""
    return {"status": "API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/classify_email", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    try:
        masker = get_pii_masker()
        model = get_classifier()
        
        email_text = request.email_body
        if not email_text or not isinstance(email_text, str):
            raise HTTPException(status_code=422, detail="Invalid email_body: Must be a non-empty string")
        
        email_text = re.sub(r'[\x00-\x1F\x7F]', ' ', email_text)
        preprocessed_text = preprocess_email_text(email_text)
        
        masked_text, entities = masker.mask_pii(preprocessed_text)
        category = model.predict(masked_text)
        
        return {
            "input_email_body": email_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_text,
            "category_of_the_email": category
        }
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")
