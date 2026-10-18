import sys
import os
import unittest

# Adiciona a pasta src ao path para importar o evaluator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluator import LLMEvaluator

class TestLLMEvaluator(unittest.TestCase):
    """Testes unitários para a classe LLMEvaluator."""
    
    def setUp(self):
        """Configuração inicial antes de cada teste."""
        self.evaluator = LLMEvaluator()
    
    def test_init_loads_env_variables(self):
        """Testa se o construtor carrega as variáveis de ambiente."""
        # Como não configuramos chaves reais no ambiente de teste,
        # esperamos que as chaves sejam None ou strings vazias
        self.assertIsInstance(self.evaluator.openai_key, (str, type(None)))
        self.assertIsInstance(self.evaluator.anthropic_key, (str, type(None)))
        self.assertIsInstance(self.evaluator.google_key, (str, type(None)))
    
    def test_evaluate_prompt_returns_list(self):
        """Testa se evaluate_prompt retorna uma lista."""
        results = self.evaluator.evaluate_prompt("Teste", providers=["google"])
        self.assertIsInstance(results, list)
    
    def test_google_error_handling(self):
        """Testa se erros da API Google são capturados corretamente."""
        # Simula uma chave inválida
        self.evaluator.google_key = "chave_invalida_teste"
        results = self.evaluator.evaluate_prompt("Teste", providers=["google"])
        
        # Deve retornar uma lista com um dicionário contendo 'error'
        self.assertEqual(len(results), 1)
        self.assertIn("error", results[0])
        self.assertIn("Google", results[0].get("error", ""))
    
    def test_skips_unconfigured_providers(self):
        """Testa se provedores sem chave são ignorados elegantemente."""
        # Força chaves vazias
        self.evaluator.openai_key = None
        self.evaluator.anthropic_key = None
        
        results = self.evaluator.evaluate_prompt("Teste", providers=["openai", "anthropic"])
        
        # Ambos devem retornar erro de "not configured"
        for result in results:
            self.assertIn("not configured", result.get("error", ""))

if __name__ == "__main__":
    unittest.main()