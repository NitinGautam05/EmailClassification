from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from utils import PIIMasker, preprocess_email_text
from models import EmailClassifier

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

pii_masker = PIIMasker()

os.makedirs("models", exist_ok=True)
model_path = "models/email_classifier.pkl"
classifier = EmailClassifier(model_path)

if not os.path.exists(model_path):
    dataset_path = "combined_emails_with_natural_pii.csv"
    classifier.train_on_real_data(dataset_path)
    classifier.save_model(model_path)

@app.post("/classify_email", response_model=EmailResponse)
async def classify_email(request: EmailRequest) -> Dict[str, Any]:
    try:
        email_text = request.email_body
        preprocessed_text = preprocess_email_text(email_text)
        masked_text, entities = pii_masker.mask_pii(preprocessed_text)
        category = classifier.predict(masked_text)
        return {
            "input_email_body": email_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_text,
            "category_of_the_email": category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}
