from transformers import XLMRobertaTokenizerFast
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import pandas as pd
from context.financial_keywords import FINANCIAL_KEYWORDS

# Load the fine-tuned model and tokenizer
model_name = "./models/fine_tuned_model"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
reader = FARMReader(model_name_or_path=model_name, use_gpu=False)

# Define document retrieval logic
def retrieve_documents():
    # Load documents from your CSV file
    df = pd.read_csv("data/rag_dataset.csv")
    documents = df.to_dict(orient="records")
    for doc in documents:
        doc['content'] = doc.pop('content')
    return documents

# Create document store and retriever
document_store = InMemoryDocumentStore()
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False,
    embed_title=True,
)

# Add documents to the document store
document_store.write_documents(retrieve_documents())
document_store.update_embeddings(retriever)

# Create a question answering pipeline using the fine-tuned model and retriever
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Example usage
def is_financial_question(question):
    for keyword in FINANCIAL_KEYWORDS:
        if keyword in question.lower():
            return True
    return False

def answer_question(question):
    if not is_financial_question(question):
        return "Maaf, saya hanya dapat membantu dalam pertanyaan finansial."
    prediction = pipeline.run(
        query=question, 
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
    )
    answer = prediction['answers'][0].answer if prediction['answers'] else "Tidak ada jawaban yang ditemukan"
    return answer

question = "Siapa itu finboost?"
answer = answer_question(question)
print(f"Q: {question}\nA: {answer}")

question = "Apa itu investasi saham?"
answer = answer_question(question)
print(f"Q: {question}\nA: {answer}")