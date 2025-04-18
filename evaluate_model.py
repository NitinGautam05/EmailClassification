from models import EmailClassifier
import pandas as pd

# Path to your dataset
dataset_path = "combined_emails_with_natural_pii.csv"

# Load dataset
df = pd.read_csv(dataset_path)
df = df.rename(columns={"email": "email_text", "type": "category"})

# Drop missing values if any
df.dropna(subset=["email_text", "category"], inplace=True)

# Initialize classifier
classifier = EmailClassifier()

# Train and evaluate
metrics = classifier.train(df, text_column="email_text", label_column="category")

# Print accuracy and classification report
print(f"\n✅ Accuracy: {metrics['accuracy']:.4f}")
print("\n📊 Detailed Classification Report:")
for label, scores in metrics["metrics"].items():
    if isinstance(scores, dict):
        print(f"{label:20s} → precision: {scores['precision']:.2f}, recall: {scores['recall']:.2f}, f1-score: {scores['f1-score']:.2f}")


