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
    default_kw = { 'class': 'input_field' }

    name = StringField(
        'ФИО',
        [ Optional() ],
        render_kw={ "placeholder": "Введите ваше ФИО", **default_kw }
    )
    
    required_field = DataRequired('Обязательное поле')

    mail = EmailField(
        'Email', 
        validators=[ 
            required_field ,
            Email('Некорректная эл. почта')
        ],
        render_kw={ "placeholder": "Введите вашу эл. почту", **default_kw  }
    )

    video_download_link = URLField(
        'Ссылка на загрузку записи',
        validators=[ required_field, valid_url ],
        render_kw={ "placeholder": "Введите ссылку прямой загрузки", **default_kw  }
    )

    gpx_data_download_link = URLField(
        'Ссылка на загрузку GPS трека',
        [ Optional(), valid_url ],
        render_kw={ "placeholder": "Введите ссылку прямой загрузки", **default_kw  }
    )
    
    focal_length = DecimalField(
        'Фокусное расстояние камеры',
        validators=[ required_field ],
        render_kw={ "placeholder": "Введите фокусное расстояние (см. раздел FAQ)", **default_kw  }
    )

    road_width = DecimalField(
        'Ширина дороги',
        validators=[ required_field ],
        render_kw={ "placeholder": "Введите ширину дороги с записи", **default_kw  }
    )

    submit = SubmitField('ОТПРАВИТЬ')
