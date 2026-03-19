# Math Problem Generator API

FastAPI-приложение для генерации математических задач с использованием LLM через OpenAI-compatible API.

### Описание
Cервис генерирует математические задачи по заданным параметрам:
- тема и подтема
- уровень сложности
- тип задачи
- количество задач
- язык (ru / en)
- наличие решения и подсказок

LLM вызывается через OpenAI-совместимый API, результат возвращается в виде строго валидного JSON.

### Запуск
```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload
```

### API
```bash
POST /generate
```
Генерация математических задач.

Пример запроса:
```json
{
  "topic": "linear_algebra",
  "subtopic": "eigenvalues",
  "difficulty": "medium",
  "problem_type": "computational",
  "num_problems": 1,
  "include_solution": true,
  "include_hints": true,
  "output_language": "ru"
}
```

Пример ответа:
```json
{
  "topic": "linear_algebra",
  "subtopic": "eigenvalues",
  "difficulty": "medium",
  "problem_type": "computational",
  "output_language": "ru",
  "problems": [
    {
      "id": 1,
      "title": "Нахождение собственных значений матрицы",
      "statement": "...",
      "final_answer": "...",
      "hints": ["..."],
      "solution_steps": ["..."]
    }
  ]
}
```
