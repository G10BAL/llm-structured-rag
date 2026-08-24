import dotenv
import os
import chromadb
from typing import List
from pydantic import Field, BaseModel
from google import genai

dotenv.load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")

chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", 8000))

gemini_client = genai.Client(api_key=API_KEY)
chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

collection = chroma_client.get_or_create_collection(
    name="customer_feedbacks",
    metadata={"hnsw:space": "cosine"}
)

raw_feedbacks = [
    "The delivery was delayed by 2 hours, and the food was cold. I'm very dissatisfied with the service!",
    "The app has a great interface; everything is intuitive and works quickly.",
    "The courier was polite, but they mixed up the order — they brought the wrong soup.",
    "The best customer support! They resolved my issue in just 5 minutes via chat."
]

def get_embedding(text: str) -> List[float]:
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values

for idx, text in enumerate(raw_feedbacks):
    embedding = get_embedding(text)
    collection.add(
        ids=[f"id_{idx}"],
        embeddings=[embedding],
        documents=[text]
    )


search_query = "problems with delivery and food"
query_embedding = get_embedding(search_query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

retrieved_docs = results["documents"][0]

for doc in retrieved_docs:
    print(f"- {doc}")


class FeedbackAnalysis(BaseModel):
    sentiment: str = Field(description="Sentiment: Positive, Negative or Neutral")
    criticality: int = Field(description="Criticality mark of a review from 1 (low) to 5 (high)")
    topics: List[str] = Field(description="Topics list (e.g.: Delivery, Quality, Service)")
    summary: str = Field(description="Short summary")

class BatchAnalysisResult(BaseModel):
    analyses: List[FeedbackAnalysis]

prompt = f"Analyze clients reviews:\n" + "\n".join(retrieved_docs)

response = gemini_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BatchAnalysisResult,
        temperature=0.1
    )
)

structured_data = BatchAnalysisResult.model_validate_json(response.text)

print("\nVALIDATED RESULT FROM GEMINI:")
print(structured_data.model_dump_json(indent=2))