import json

import requests


class TelegramLogger(object):

    def __init__(self, bot_token: str, chat_id: str, disable_req_log: bool = True):
        self._api_url = f'https://api.telegram.org/bot{bot_token}/'
        self._message_url = self._api_url + 'sendMessage'
        self._image_url = self._api_url + 'sendMediaGroup'
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

    def log_images(self, image_paths, message: str = ''):
        try:
            send_data = {
                'chat_id': self._chat_id,
                'media': json.dumps([
                    {
                        'type': 'photo',
                        'media': f'attach://image_{i}.png',
                        'caption': message if i == 0 else '',
                    } for i in range(len(image_paths))
                ])
            }
            files = {f'image_{i}.png': open(image_path, 'rb') for i, image_path in enumerate(image_paths)}
            requests.post(self._image_url, params=send_data, files=files)
        except Exception:
            pass
