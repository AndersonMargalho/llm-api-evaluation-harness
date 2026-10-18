import csv
import os

def save_to_csv(results: list, filename: str = "evaluation_results.csv"):
    """
    Salva os resultados da avaliação em um arquivo CSV.
    """
    # Define as colunas da planilha
    headers = ["provider", "model", "response", "tokens_used", "latency_seconds", "cost_usd"]
    
    # Garante que a pasta de saída existe (opcional, mas boa prática)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for result in results:
                # Se der erro na API, ainda salvamos o erro para registro
                if "error" in result:
                    writer.writerow({
                        "provider": result.get("provider", "Unknown"),
                        "model": result.get("model", "Unknown"),
                        "response": f"ERROR: {result['error']}",
                        "tokens_used": 0,
                        "latency_seconds": 0,
                        "cost_usd": 0
                    })
                else:
                    writer.writerow(result)
                    
        print(f"\n✅ Resultados salvos com sucesso em: {filepath}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar CSV: {e}")