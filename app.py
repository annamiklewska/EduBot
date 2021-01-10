"""
This bot listens to port 5002 for incoming connections from Facebook. It takes
in any messages that the bot receives and echos it back.
"""
from flask import Flask, request
from pymessenger.bot import Bot

app = Flask(__name__)

ACCESS_TOKEN = "EAAGPgq0vyAUBAGfdf0XIDpX5HrLXw0Pdu0MFXOxX50IQkO79o2OH06eLckviBJxih6cj4KLmvZABeanTX3P0l4rVxBvC25GFHn5R0ZBHoTj1QMt7LjLfSK51Ei1wcOYw9DZAS9jQuZA44eJejhqsQLZAwB0IC99c0aDlavL2EYRgWHvbmURMiZCIPFGuVb3xgZD"
VERIFY_TOKEN = "456f4873rfre67g4fds5g54/t5432tfd564sgdfs8t9434t36t1g564g2329yrhgsiutryg348t693tntrh4574"
bot = Bot(ACCESS_TOKEN)


@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        if request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return request.args.get("hub.challenge")
        else:
            return 'Invalid verification token'

    if request.method == 'POST':
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for x in messaging:
                if x.get('message'):
                    recipient_id = x['sender']['id']
                    if x['message'].get('text'):
                        message = x['message']['text']
                        print(message)
                        print(bot.send_text_message(recipient_id, message))
                    if x['message'].get('attachments'):
                        print("pass")

                        #for att in x['message'].get('attachments'):
                        #    bot.send_attachment_url(recipient_id, att['type'], att['payload']['url'])
                else:
                    pass
        return "Success"


if __name__ == "__main__":
    app.run()
