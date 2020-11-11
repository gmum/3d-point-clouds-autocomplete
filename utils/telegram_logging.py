import requests


class TelegramLogger(object):

    def __init__(self, bot_token: str, chat_id: str, disable_req_log: bool = True):
        self._api_url = f'https://api.telegram.org/bot{bot_token}/'
        self._message_url = self._api_url + 'sendMessage'
        self._image_url = self._api_url + 'sendPhoto'
        self._chat_id = chat_id

        if disable_req_log:
            import logging
            logging.getLogger("requests").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    def log(self, message: str):
        try:
            send_data = {
                'chat_id': self._chat_id,
                'text': message,
            }
            requests.post(self._message_url, json=send_data)
        except Exception:
            pass

    def log_images(self, image_pathes):
        try:
            files = {'image' + str(i): open(image_path, 'rb') for i, image_path in enumerate(image_pathes)}
            data = {'chat_id': self._chat_id}
            requests.post(self._image_url, files=files, data=data)

        except Exception:
            pass
