import requests


class TelegramLogger(object):

    def __init__(self, bot_token: str, chat_id: str, disable_req_log: bool = True):
        self._api_url = f'https://api.telegram.org/bot{bot_token}/'
        self._message_url = self._api_url + 'sendMessage'
        self._chat_id = chat_id

        if disable_req_log:
            import logging
            logging.getLogger("requests").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)

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
