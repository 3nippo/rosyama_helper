import os
import subprocess
from src.analyze_damage import damage_items_empty
from geopy.geocoders import Nominatim


geolocator = Nominatim(user_agent="app")


eng_label2rus_label = {
    'pothole': 'яма',
    'water-puddle': 'яма',
    'cracks': 'трещины',
    'unpaved': 'без покрытия'
}


def generate_report(
    damage_list, 
    track, 
    report_dir,
    frames_dir,
    request_data
):
    mdpath = os.path.join(report_dir, 'report.md')

    generate_md_report(
        damage_list,
        track,
        mdpath,
        frames_dir,
        request_data
    )

    return convert_md_to_pdf(mdpath)


def generate_md_report(
    damage_list, 
    track, 
    report_path,
    frames_dir,
    request_data
):
    with open(report_path, 'w') as report:
        quote = False
        
        def rprint(*args):
            if quote:
                print('>', *args, '  ', file=report)
                print('>', file=report)
            else:
                print(*args, '  ', file=report)
                print(file=report)

        rprint("# Отчет о повреждениях дороги")

        rprint('## ФИО пользователя:', request_data['name'])

        rprint('## Почта пользователя:', '<{}>'.format(request_data['mail']))
        
        for i in range(len(damage_list)):
            rprint('***')

            rprint('### Участок дороги №{}'.format(i+1))

            if damage_items_empty(damage_list[i]):
                rprint('#### Повреждений не обнаружено')
                continue

            point, _ = track[i]
            
            rprint('#### Местоположение')

            quote = True

            rprint("Широта:", point[0])
            rprint("Долгота:", point[1])
            
            address = geolocator.reverse(point).address

            rprint("Адрес:", address)

            quote = False

            for j, damage_item in enumerate(damage_list[i]):
                rprint('#### Повреждение №{}'.format(j+1))
                
                quote = True

                rprint('Тип повреждения:', eng_label2rus_label[damage_item.type])
                
                rprint('Площадь повреждения:', damage_item.area, 'м^2')

                if damage_item.type == 'unpaved':
                    quote = False
                    
                    continue

                rprint('Ширина повреждения:', damage_item.width, 'м')
                
                rprint('Длина повреждения:', damage_item.length, 'м')

                quote = False

            rprint('![Участок дороги {}]({})'.format(
                i+1, 
                os.path.join(
                    os.path.abspath(frames_dir),
                    '{}.jpg'.format(i)
                )
            ))


def convert_md_to_pdf(mdpath):
    pdfpath = mdpath.replace('.md', '.pdf')

    convert_cmd = "markdown-pdf -o {} {}".format(pdfpath, mdpath)
    
    process = subprocess.Popen(
        convert_cmd.split(),
        stderr=subprocess.PIPE
    )
    
    _, stderr = process.communicate()
    
    if stderr:
        raise RuntimeError

    return pdfpath
