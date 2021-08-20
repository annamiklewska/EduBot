"""
This bot listens to port 5002 for incoming connections from Facebook. It takes
in any messages that the bot receives and echos it back.
"""
import os

from flask import Flask, request
from pymessenger.bot import Bot

app = Flask(__name__)

ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
VERIFY_TOKEN = os.environ['VERIFY_TOKEN']
bot = Bot(ACCESS_TOKEN)


@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        if request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return request.args.get("hub.challenge")
        else:
            return 'Invalid verification token'

    if request.method == 'POST':
        output = request.data
        print(output)
        '''
        for event in output['entry']:
            messaging = event['messaging']
            for x in messaging:
                if x.get('message'):
                    recipient_id = x['sender']['id']
                    if x['message'].get('text'):
                        message = x['message']['text']
                        message = "test msg!"
                        print(message)
                        print(bot.send_text_message(recipient_id, message))
                    if x['message'].get('attachments'):
                        print("pass")

                        #for att in x['message'].get('attachments'):
                        #    bot.send_attachment_url(recipient_id, att['type'], att['payload']['url'])
                else:
                    pass'''
        return "Success"


if __name__ == "__main__":
    app.run()
