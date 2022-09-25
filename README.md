# Описание проекта
Решает задачу по определению SMS-спама, данные взяты из соревнования https://www.kaggle.com/uciml/sms-spam-collection-dataset

Реализовано:
- Пайплайн обучения ML-модели с использованием DVC
- Сервис на flask с методами:
    - /forward, для прогона единственного сообщения
    - /forward_batch, для прогона нескольких сообщений сразу
    - /evaluate, для измерения качества ML-модели на переданном датасете
    - /add_data, для добавления данных в train
    - /retrain, для переобучения текущей модели
    - /deploy/<experiment_id> для замены модели на модель из experiment_id
    - /metrics/<experiment_id> для получения метрики модели из experiment_id

# Запуск

`python3 app.py`

# Примеры методов

Перед запуском методов ниже рекомендуется вызвать `dvc repro` для загрузки всех нужных данных для их работы.

## CURL /forward OK
`curl -X POST http://localhost:9002/forward -d '{"message": "John go walk"}' -H "Content-Type: application/json"`

## CURL /forward_batch 200
`curl -X POST http://localhost:9002/forward_batch -H "Content-Type:multipart/form-data" -F df=@data/test_data.tsv`

## CURL /evaluate 200
`curl -X POST http://localhost:9002/evaluate -H "Content-Type:multipart/form-data" -F df=@data/test_data.tsv`

## CURL /add_data 200
`curl -X PUT http://localhost:9002/add_data -H "Content-Type:multipart/form-data" -F df=@data/test_data.tsv`

## CURL /retrain 200
`curl -X PUT http://localhost:9002/retrain`

## CURL /deploy/<experiment_id> 200
`curl http://localhost:9002/deploy/exp_09-26-2022_01-35-51`

## CURL /metrics/<experiment_id> 200
`curl http://localhost:9002/deploy/exp_09-26-2022_01-35-51`