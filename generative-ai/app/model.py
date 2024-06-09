from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class FinancialIndoGPT:
    def __init__(self, model_name="cahya/gpt2-small-indonesian-522M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, question, max_length=150):
        inputs = self.tokenizer.encode(question, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

gpt_model = FinancialIndoGPT()
