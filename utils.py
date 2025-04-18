import re
import spacy
import json
from typing import Dict, List, Tuple, Any

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_md")
except:
    # If model isn't downloaded, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

class PIIMasker:
    def __init__(self):
        # Define regex patterns for different PII/PCI entities
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "dob": r'\b(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}\b',
            "aadhar_num": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "credit_debit_no": r'\b(?:\d{4}[\s-]?){4}\b',
            "cvv_no": r'\bCVV:?\s*\d{3,4}\b',
            "expiry_no": r'\b(0?[1-9]|1[0-2])[\/\-]\d{2,4}\b'
        }
    
    def detect_names(self, text: str) -> List[Dict[str, Any]]:
        """Detect full names using spaCy NER"""
        doc = nlp(text)
        names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.append({
                    "position": [ent.start_char, ent.end_char],
                    "classification": "full_name",
                    "entity": ent.text
                })
        return names
    
    def detect_pii_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII/PCI entities using regex patterns"""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "position": [match.start(), match.end()],
                    "classification": entity_type,
                    "entity": match.group()
                })
        
        return entities
    
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask all PII entities in the text and return masked text and entity list
        """
        # Get all entities
        regex_entities = self.detect_pii_with_regex(text)
        name_entities = self.detect_names(text)
        all_entities = regex_entities + name_entities
        
        # Sort entities by their start position in descending order
        # This ensures we replace from end to start to maintain correct positions
        all_entities.sort(key=lambda x: x["position"][0], reverse=True)
        
        # copy of the original text
        masked_text = text
        
        # Mask each entity
        for entity in all_entities:
            start, end = entity["position"]
            entity_type = entity["classification"]
            masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
        
        # Re-detect entities with correct positions in the original text
        all_entities = regex_entities + name_entities
        all_entities.sort(key=lambda x: x["position"][0])
        
        return masked_text, all_entities

def preprocess_email_text(email_text: str) -> str:
    """
    Basic preprocessing for email text:
    - Remove email headers if present
    - Normalize whitespace
    - Remove excessive newlines
    """
    # Remove any potential email headers
    if "\nSubject:" in email_text:
        email_text = email_text.split("\n\n", 1)[-1]
    
    # Normalize whitespace
    email_text = re.sub(r'\s+', ' ', email_text)
    
    # Strip leading/trailing whitespace
    email_text = email_text.strip()
    
    return email_text