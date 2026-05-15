# Fine-tuning Pipeline

В данной части проекта реализован пайплайн fine-tuning модели для задачи классификации тональности текста.  
Fine-tuning выполняется с использованием PEFT-подхода LoRA.

## 1. Задача

Выбрана задача sentiment classification из области Natural Language Processing.

Цель задачи — по тексту отзыва на фильм определить его тональность:

- `0` — отрицательный отзыв;
- `1` — положительный отзыв.

Пример входных и выходных данных:

```json
{
  "text": "This movie was surprisingly good...",
  "label": 1
}
```

Входом модели является поле text, выходом — бинарная метка label.

## 2. Набор данных
Для экспериментов используется датасет:
```txt
stanfordnlp/imdb
```
Его подвыборка:

| Split      | Размер |
|------------|--------|
| train      | 5000   |
| validation | 1000   |
| test       | 1000   |

Датасет содержит отзывы на фильмы и бинарные метки тональности.
Подготовленные данные сохраняются в директорию:
```txt
finetuning/data/imdb/
```

## 3. Стратегия экспериментов
Выбрана стратегия прогрессии.

Сначала запускается более легкая модель, затем более крупная и сильная:

1. distilbert-base-uncased
2. bert-base-uncased
3. roberta-base

Такой порядок позволяет сначала проверить работоспособность пайплайна на дешевом эксперименте, а затем сравнить качество более сильных моделей.

Дополнительно для distilbert-base-uncased проведены эксперименты с разными значениями ранга LoRA:
- r = 4
- r = 8
- r = 16

Это позволяет оценить влияние гиперпараметра lora_r на качество и размер адаптера.

## 4. Метод fine-tuning
Для fine-tuning используется LoRA из библиотеки Hugging Face PEFT.

Базовая модель остается замороженной, а обучаются только небольшие LoRA-адаптеры. Это позволяет уменьшить количество обучаемых параметров и выполнить обучение на ограниченных вычислительных ресурсах.

QLoRA не использовал, так как выполняю на arm.

## 5. Установка зависимостей
```bash
pip install torch transformers datasets peft accelerate evaluate scikit-learn pandas matplotlib
```

## 6. Подготовка данных
Для подготовки датасета нужно запустить:
```bash
python -m finetuning.prepare_dataset
```

## 7. Запуск одного эксперимента
Пример запуска LoRA fine-tuning для distilbert-base-uncased:
```bash
python -m finetuning.train_lora_classifier \
  --run-name 01_distilbert_lora_r8 \
  --model-name distilbert-base-uncased \
  --target-modules q_lin,v_lin \
  --epochs 1 \
  --batch-size 4 \
  --max-length 128 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05
```

## 8. Запуск всех экспериментов
Для запуска всех экспериментов по выбранной стратегии:
```bash
python -m finetuning.run_experiments
```

Список экспериментов:
| Run                    | Модель                  | LoRA r | LoRA alpha |
| ---------------------- | ----------------------- | -----: | ---------: |
| 01_distilbert_lora_r8  | distilbert-base-uncased |      8 |         16 |
| 02_bert_lora_r8        | bert-base-uncased       |      8 |         16 |
| 03_roberta_lora_r8     | roberta-base            |      8 |         16 |
| 04_distilbert_lora_r4  | distilbert-base-uncased |      4 |          8 |
| 05_distilbert_lora_r16 | distilbert-base-uncased |     16 |         32 |

## 9. Сбор результатов
После завершения обучения нужно собрать итоговую таблицу и график цена/качество:
```bash
python -m finetuning.collect_results
```
После запуска создаются файлы:

```txt
finetuning/outputs/summary.csv
finetuning/outputs/price_quality_curve.png
```
Для каждого эксперимента сохраняются:

- metrics.json — метрики качества;
- experiment_config.json — конфигурация модели и гиперпараметров;
- adapter/ — сохранённый LoRA-адаптер;
- checkpoint-* — контрольные точки обучения.

## 10. Метрики
Для оценки качества используются: 
- accuracy; 
- F1-score; 
- eval_loss.

Для оценки цены эксперимента используются:

- время обучения;
- размер LoRA-адаптера;
- количество обучаемых параметров;
- скорость обучения.

## 11. Полученные результаты
Итоговые результаты экспериментов:
| Run                    | Модель                  | LoRA r | Accuracy |     F1 | Время, сек | Размер адаптера, MB |
| ---------------------- | ----------------------- | -----: | -------: | -----: | ---------: | ------------------: |
| 01_distilbert_lora_r8  | distilbert-base-uncased |      8 |    0.844 | 0.8497 |      60.14 |                3.51 |
| 02_bert_lora_r8        | bert-base-uncased       |      8 |    0.850 | 0.8571 |     159.42 |                1.83 |
| 03_roberta_lora_r8     | roberta-base            |      8 |    0.895 | 0.8953 |     156.38 |                6.80 |
| 04_distilbert_lora_r4  | distilbert-base-uncased |      4 |    0.837 | 0.8428 |      56.42 |                3.23 |
| 05_distilbert_lora_r16 | distilbert-base-uncased |     16 |    0.844 | 0.8503 |      56.03 |                4.08 |


Лучшее качество показала модель:
```txt
roberta-base + LoRA r=8
```
Результаты:
```txt
Accuracy = 0.895
F1 = 0.8953
```

