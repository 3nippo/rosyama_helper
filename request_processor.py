import sys
import os
import pika
import json 
import functools

from src.obtain_masks import load_model
from src.analyze_damage import load_pydict
from src.process_request import process_request, load_name_label_dicts, BadError, CantDownloadError

"""
request.json

{
    'name': "Name",
    'mail': 'some_mail@address.com',
    'video_download_link': 'https://www.link.com/my_video.mp4',
    '(optional)gpx_data_download_link': 'https://www.link.com/my_track.gpx',
    'focal_length': 35,
    'road_width': 6
}
"""

def process_message(channel, frame, properties, body, model, name2label, label2name, gmail_login):
    request_data = json.loads(body)
    
    try:
        process_request(
            model, 
            name2label, 
            label2name,
            gmail_login,
            request_data
        )
    except (CantDownloadError, BadError) as exception:
        print(str(exception))

    channel.basic_ack(delivery_tag=frame.delivery_tag)

def run_consumer_logic(config, on_message_cb):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(port=config['MQ_PORT'])
    )

    channel = connection.channel()
    
    channel.exchange_declare(
        exchange=config['EXCHANGE_NAME'],
        exchange_type=config['EXCHANGE_TYPE'],
        durable=True,
        passive=True
    )

    channel.queue_declare(
        queue=config['QUEUE_NAME'],
        durable=True
    )

    channel.queue_bind(
        queue=config['QUEUE_NAME'],
        exchange=config['EXCHANGE_NAME']
    )

    channel.basic_consume(
        queue=config['QUEUE_NAME'], 
        on_message_callback=on_message_cb
    )

    channel.basic_qos(
        prefetch_count=1
    )
    
    print('Started consumption')
    channel.start_consuming()

if __name__ == '__main__':
    config = None

    with open('config.json') as config_file:
        config = json.loads(config_file.read())

    model = load_model(config['MODEL_DIR'])
    
    name2label, label2name = load_name_label_dicts(config['DICTS_DIR'])

    mq_port = config['MQ_PORT']
    queue_name = config['QUEUE_NAME']
    
    on_message_cb = functools.partial(
        process_message,
        model=model,
        name2label=name2label,
        label2name=label2name,
        gmail_login=config['GMAIL_LOGIN']
    )

    try:
        run_consumer_logic(
            config,
            on_message_cb
        )
    except KeyboardInterrupt:
        print('Interrupted')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
