import requests

import logging
logging.getLogger("requests").setLevel(logging.CRITICAL)


class TelegramLogger(object):

    def __init__(self, bot_token: str, chat_id: str):
        self._api_url = f'https://api.telegram.org/bot{bot_token}/'
        self._message_url = self._api_url + 'sendMessage'
        self._chat_id = chat_id

    def log(self, message: str):
        try:
            self.__send_message_to_user(message)
        except Exception:
            pass

    def __send_message_to_user(self, message: str):
        send_data = {
            "chat_id": self._chat_id,
            "text": message,
        }
        requests.post(self._message_url, json=send_data)
