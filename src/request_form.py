import json 
import decimal

from flask_wtf import FlaskForm

from wtforms.fields import StringField, SubmitField
from wtforms.fields.html5 import EmailField, URLField, DecimalField

from wtforms.validators import DataRequired, Email, Optional, ValidationError

import validators

ignore_fields = set([
    'submit', 
    'csrf_token'
])


def valid_url(form, field):
    if (
        field.data 
        and not validators.url(field.data)
    ):
        raise ValidationError("Некорректная ссылка")


class RequestEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


class RequestForm(FlaskForm):
    name = StringField(
        'ФИО',
        [ Optional() ]
    )
    
    required_field = DataRequired('Обязательное поле')

    mail = EmailField(
        'Email', 
        validators=[ 
            required_field ,
            Email('Некорректная эл. почта')
        ]
    )

    video_download_link = URLField(
        'Ссылка на загрузку записи',
        validators=[ required_field, valid_url ]
    )

    gpx_data_download_link = URLField(
        'Ссылка на загрузку GPS трека',
        [ Optional(), valid_url ]
    )
    
    focal_length = DecimalField(
        'Фокусное расстояние камеры',
        validators=[ required_field ]
    )

    road_width = DecimalField(
        'Ширина дороги',
        validators=[ required_field ]
    )

    submit = SubmitField('Отправить')
