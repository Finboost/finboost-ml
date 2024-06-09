from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class FinancialIndoGPT:
    def __init__(self, model_name="cahya/gpt2-small-indonesian-522M"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, question, max_length=150):
        inputs = self.tokenizer.encode(question, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        filtered_response = self.filter_response(response)
        return filtered_response

    def filter_response(self, response):
        # Filtering logic to ensure response relevance
        financial_keywords = ["uang", "investasi", "perbankan", "kredit", "pasar modal", "asuransi", "keuangan", "manajemen keuangan"]
        sentences = response.split(".")
        filtered_sentences = [sentence.strip() for sentence in sentences if any(keyword in sentence for keyword in financial_keywords)]
        filtered_response = ". ".join(filtered_sentences).strip()

        if not filtered_response:
            return "Maaf, saya tidak memiliki informasi yang relevan tentang pertanyaan Anda."
        
        return filtered_response

gpt_model = FinancialIndoGPT()
