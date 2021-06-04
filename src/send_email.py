import yagmail


def send_email(
    gmail_login,
    request_data,
    report_path
):
    name = request_data['name'] or request_data['mail'].split('@')[0]

    contents = "Доброго дня, {}!\n\nОтчет о поврежденности дороги во вложениях.".format(name)

    yagmail.SMTP(gmail_login).send(
        to=request_data['mail'],
        subject='Отчет о поврежденности дороги',
        contents=contents,
        attachments=[report_path]
    )
