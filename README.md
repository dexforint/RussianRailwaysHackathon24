# Решение хакатона

![Лого Паровозика](/public/logo.png)

## Описание

Решение представляет собой телеграм-бот, в который пользователь (сотрудник РЖД) может зайти и задать свой вопрос по работе в компании ОАО «РЖД». Данный сервис позволит съэкономить время пользователю, ищя за него всю необходимую информации в большой документации компании.

## Технические особенности

Язык программирования Python, telegram бот (aiogram), LLM, RAG.

## Особенности

Особенность нашего решения состоит в том, что наш сервис предоставляет точные ответы на вопросы и удобство для пользователя, позволяя искусственному интеллекту делать за него всю "грязную" работу.

## Команда

Паровозик, который смог

## Демо

https://t.me/ParovozicBot

## Установка и запуск

Язык программирования: **Python (3.12)**

1. Установка зависимостей:

```bash
pip install requirements.txt
```

2. Скопируйте файл .env.example и переименуйте его в `.env`
3. Создайте телеграм бота в https://t.me/BotFather, скопируйте токен
4. В самом файле `.env` вставте токен для значения `TELEGRAM_TOKEN`
5. Запустите бота

```bash
python bot.py
```

## Автор

https://t.me/dl_hello
