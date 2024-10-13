"""
Telegram бот для конвертации голосового/аудио сообщения в текст и
создания аудио из текста.
"""

import logging
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.types.input_file import FSInputFile
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.types import Message, CallbackQuery
from aiogram import Router
from dotenv import load_dotenv

from aiogram.types import Update
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from stt import audio_to_text
from task_tools import generate_answer

import os

logging.basicConfig(level=logging.INFO)


load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()


database = {}

router = Router()

# Словарь, который будет хранить роли пользователей
user_id2role = {}

# Доступные роли
roles = ["простой сотрудник", "машинист", "инженер", "оператор", "менеджер", "кадровик"]


# Команда /role для выбора роли
@router.message(Command("role"))
async def choose_role(message: Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=role, callback_data=role)] for role in roles
        ]
    )
    await message.answer(
        f"Ваша текущая роль: {user_id2role[message.from_user.id]} \n Выберите свою роль:",
        reply_markup=keyboard,
    )


# Обработчик выбора роли
@router.callback_query(F.data.in_(roles))
async def role_callback(callback: CallbackQuery):
    # Получаем ID пользователя и выбранную роль
    user_id = callback.from_user.id
    selected_role = callback.data

    # Сохраняем роль в словарь
    user_id2role[user_id] = selected_role

    # Подтверждение выбора
    await callback.message.edit_text(f"Ваша роль была установлена как: {selected_role}")

    # Уведомляем, что кнопка была обработана
    await callback.answer()


# Главное меню с кнопками для команд
def main_menu():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/start"), KeyboardButton(text="/role")],
        ],
        resize_keyboard=True,
    )


# Обработчик команд /start и /help
@router.message(Command(commands=["start", "help"]))
async def cmd_start_and_help(message: Message):
    # user_id = message.from_user.id
    await message.answer(
        f"""Привет! Я Паровозик - чат-бот, который поможет тебе ответить на вопросы по документации ОАО "РЖД" !\n\n'
Чтобы получить ответ на свой вопрос достаточно просто ввести вопрос текстом или голосом.
Вы так же можете выбрать свою роль в компании для персонализированного ответа.

/start - Начать работу
/role - Выбрать свою роль
""",
        reply_markup=main_menu(),  # Отправляем главное меню
    )


@router.message(F.voice)
async def handle_voice(message: Message):

    voice_file_id = message.voice.file_id
    voice_file = await bot.get_file(voice_file_id)

    file_path = voice_file.file_path
    await message.answer(
        f"Я получил голосовое сообщение с file_id: {voice_file_id}.\nПуть к файлу: {file_path}"
    )

    # Скачивание голосового сообщения
    voice = await bot.download_file(file_path)
    file_path = f"./tmp/voice_{message.from_user.id}.ogg"
    with open(file_path, "wb") as f:
        f.write(voice.read())

    query = audio_to_text(file_path)

    user_id = message.from_user.id
    user_role = user_id2role[user_id]

    answer = generate_answer(query, f"Роль пользователя в компании: {user_role}")

    await message.answer(answer)


@router.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    user_role = user_id2role[user_id]

    query = message.text

    answer = generate_answer(query, f"Роль пользователя в компании: {user_role}")

    await message.answer(answer)


# Регистрация роутера в диспетчере
dp.include_router(router)


# Middleware для сбора статистики
class UserStatisticsMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: Update, data: dict[str, any]) -> any:
        user = event.from_user
        if not (user.id in user_id2role):
            user_id2role[user.id] = "простой сотрудник"

        if isinstance(event, Message):

            logging.info(
                f"Пользователь {user.id} ({user.first_name}) отправил сообщение: {event.text}"
            )
        elif isinstance(event, CallbackQuery):
            logging.info(
                f"Пользователь {user.id} ({user.first_name}) нажал на кнопку: {event.data}"
            )

        # Не блокируем следующие обработчики, передаем выполнение дальше
        return await handler(event, data)


# Регистрация middleware для сбора статистики
dp.message.middleware(UserStatisticsMiddleware())
dp.callback_query.middleware(UserStatisticsMiddleware())


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
