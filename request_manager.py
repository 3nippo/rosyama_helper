import pika
import json 

from flask import Flask, render_template, flash, request, jsonify
import src.request_form as request_form

app = Flask(__name__)

config = None


def send_request_data(request_data):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(port=config['MQ_PORT'])
    )

    channel = connection.channel()

    channel.basic_qos(
        prefetch_count=1
    )

    channel.basic_recover(requeue=True)

    channel.exchange_declare(
        exchange=config['EXCHANGE_NAME'],
        exchange_type=config['EXCHANGE_TYPE'],
        durable=True
    )

    channel.queue_declare(
        queue=config['QUEUE_NAME'],
        durable=True
    )

    channel.queue_bind(
        queue=config['QUEUE_NAME'],
        exchange=config['EXCHANGE_NAME']
    )
    
    channel.basic_publish(
        exchange=config['EXCHANGE_NAME'],
        routing_key=config['QUEUE_NAME'],
        body=request_data
    )

    channel.close()

    connection.close()


@app.route("/", methods=['GET', 'POST'])
def home():
    form = request_form.RequestForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        flash("Заявка отправлена!")
        
        form_data = dict(form.data)

        for ignore_field in request_form.ignore_fields:
            form_data.pop(ignore_field, None)

        form_data = json.dumps(form_data, cls=app.json_encoder)

        send_request_data(form_data)

    return render_template('make_request.html', form=form)

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.loads(config_file.read())
    
    app.config['SECRET_KEY'] = config['SECRET_KEY']
    
    app.json_encoder = request_form.RequestEncoder

    app.run(port=1961)
