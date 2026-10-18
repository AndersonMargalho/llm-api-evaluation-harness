import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMEvaluator:
    """Avalia múltiplos provedores LLM via API direta."""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def call_openai(self, prompt: str, model: str = "gpt-4o-mini") -> dict:
        if not self.openai_key:
            return {"error": "OpenAI API key not configured"}
        
        start_time = time.time()
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500}
            )
            elapsed = time.time() - start_time
            data = response.json()
            
            if "error" in data:
                return {"error": f"OpenAI Error: {data['error'].get('message', 'Unknown')}"}
                
            return {
                "provider": "OpenAI", "model": model,
                "response": data["choices"][0]["message"]["content"],
                "tokens_used": data["usage"]["total_tokens"],
                "latency_seconds": round(elapsed, 2),
                "cost_usd": self._calculate_openai_cost(data["usage"]["total_tokens"], model)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def call_anthropic(self, prompt: str, model: str = "claude-3-haiku-20240307") -> dict:
        if not self.anthropic_key:
            return {"error": "Anthropic API key not configured"}
        
        start_time = time.time()
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.anthropic_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 500, "messages": [{"role": "user", "content": prompt}]}
            )
            elapsed = time.time() - start_time
            data = response.json()
            
            if "error" in data:
                return {"error": f"Anthropic Error: {data['error'].get('message', 'Unknown')}"}
                
            return {
                "provider": "Anthropic", "model": model,
                "response": data["content"][0]["text"],
                "tokens_used": data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                "latency_seconds": round(elapsed, 2),
                "cost_usd": self._calculate_anthropic_cost(data["usage"], model)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def call_google(self, prompt: str, model: str = "gemini-3.8-flash") -> dict:
        if not self.google_key:
            return {"error": "Google API key not configured"}
        
        start_time = time.time()
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.google_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            elapsed = time.time() - start_time
            data = response.json()
            
            if "error" in data:
                return {"error": f"Google API Error: {data['error'].get('message', 'Unknown error')}"}
                
            return {
                "provider": "Google", "model": model,
                "response": data["candidates"][0]["content"]["parts"][0]["text"],
                "tokens_used": 0,
                "latency_seconds": round(elapsed, 2),
                "cost_usd": 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_openai_cost(self, tokens: int, model: str) -> float:
        prices = {"gpt-4o-mini": 0.15, "gpt-4o": 2.50}
        return round((tokens / 1_000_000) * prices.get(model, 0.15), 6)
    
    def _calculate_anthropic_cost(self, usage: dict, model: str) -> float:
        prices = {"claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}}
        model_prices = prices.get(model, {"input": 0.25, "output": 1.25})
        input_cost = (usage["input_tokens"] / 1_000_000) * model_prices["input"]
        output_cost = (usage["output_tokens"] / 1_000_000) * model_prices["output"]
        return round(input_cost + output_cost, 6)
    
    def evaluate_prompt(self, prompt: str, providers: list = None) -> list:
        if providers is None:
            providers = ["openai", "anthropic", "google"]
        
        results = []
        if "openai" in providers:
            results.append(self.call_openai(prompt))
        if "anthropic" in providers:
            results.append(self.call_anthropic(prompt))
        if "google" in providers:
            results.append(self.call_google(prompt))
        
        return results

if __name__ == "__main__":
    # Importa a função de salvar CSV que criamos no utils.py
    from utils import save_to_csv
    
    evaluator = LLMEvaluator()
    test_prompt = "Explique o que é machine learning em uma frase simples."
    
    print("Avaliando prompt...\n")
    results = evaluator.evaluate_prompt(test_prompt)
    
    for result in results:
        if "error" in result:
            print(f" ERRO no {result.get('provider', 'Unknown')}: {result['error']}")
        else:
            print(f"Provider: {result.get('provider')}")
            print(f"Model: {result.get('model')}")
            print(f"Response: {result.get('response')[:100]}...")
            print(f"Latency: {result.get('latency_seconds')}s")
            print(f"Cost: ${result.get('cost_usd')}")
        print("-" * 50)
        
    # Chama a função para gerar a planilha
    save_to_csv(results)