import cv2
import asyncio
import numpy as np
import tensorflow as tf
from telegram.ext import MessageHandler, filters, Application, CommandHandler

checkpoint_path = "/Users/ajaydyavathi/Documents/FinalProjects/Priyanka UCM/CIFAR-10-files/model_best.h5"
model4 = tf.keras.models.load_model(checkpoint_path)

ACCESS_TOKEN = "6899100680:AAEgo0foILi5qSv91d-xbfJOTAr8Plln9ro"

label_meta_data = ['airplane',
                   'automobile',
                   'bird',
                   'cat',
                   'deer',
                   'dog',
                   'frog',
                   'horse',
                   'ship',
                   'truck']

async def start(update, context):
    intro = "Hello, I'm a telegram bot capable of classifying images."
    return await context.bot.send_message(chat_id=update.effective_chat.id, text=intro)


async def about(update, context):
    about_text = "This is a CIFAR-10 image classification bot"
    return await context.bot.send_message(chat_id=update.effective_chat.id, text=about_text)


async def perform_classification(update, context):
    image_file = await update.message.photo[-1].get_file()
    uid = update.effective_user.id
    save_name = "{}.png".format(uid)
    await image_file.download_to_drive(save_name)
    image = cv2.resize(cv2.imread(save_name), (32, 32)) / 255.0
    tensor_image = tf.convert_to_tensor(np.expand_dims(image, 0))
    prediction = model4.predict(tensor_image).argmax(1)[0]
    prediction_label = label_meta_data[prediction]
    return await context.bot.send_message(chat_id=update.effective_chat.id, text="It seems like a/an {}.".format(prediction_label))


def main():
    cifar_app = Application.builder().token(ACCESS_TOKEN).build()
    cifar_app.add_handler(CommandHandler("start", start))
    cifar_app.add_handler(CommandHandler("about", about))
    cifar_app.add_handler(MessageHandler(filters.PHOTO, perform_classification))
    cifar_app.run_polling()


if __name__ == "__main__":
    main()
