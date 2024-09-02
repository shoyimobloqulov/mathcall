import os
from django.conf import settings
from django.shortcuts import redirect, render # type: ignore
from django.http import HttpResponse # type: ignore
import pdfplumber # type: ignore
import re
from django.template import loader # type: ignore

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.special import gamma # type: ignore

from django.shortcuts import render # type: ignore
from django.http import HttpResponse # type: ignore
from .forms import CalculationForm, SimulationForm
import csv
from io import BytesIO
from .tasks import perform_calculation
from django.http import JsonResponse # type: ignore
from scipy.special import gamma # type: ignore
import json

import csv
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
import logging
from multiprocessing import Pool
logger = logging.getLogger(__name__)

def introduction(request):
    return render(request, 'introduction.html')

items = [
    {'id': '1', 'name': 'Аномальный перенос вещества в двухзонной фрактальной среде', 'file': 'pdf_files/Аномальный перенос вещества в двухзонной фрактальной среде.pdf'},
    {'id': '2', 'name': 'Аномальный перенос с учетом адсорбционных эффектов и разложения вещества', 'file': 'pdf_files/Аномальный перенос с учетом адсорбционных эффектов и разложения вещества.pdf'},
    {'id': '3', 'name': 'Аномальный перенос с много – членными дробными производными', 'file': 'pdf_files/Аномальный перенос с много – членными дробными производными.pdf'},
]
def code(request):
    return render(request, 'code.html')
def published_works_list(request):
    works = [
        {
            "url": "(tezis 1) Termiz-2022_Ç¬Γπá½. ó«»α«ßδ á½úÑíαδ ¿ á¡á½¿ºá ßí«α¡¿¬ ¼áΓÑα¿á½«ó αÑß». ¡áπτ¡«-»αá¬. ¬«¡ΣÑαÑ¡µ¿¿.pdf",
            "name": " АЛГЕБРА ВА АНАЛИЗНИНГ ДОЛЗАРБ МАСАЛАЛАРИ МАВЗУСИДАГИ РЕСПУБЛИКА ИЛМИЙ-АМАЛИЙ АНЖУМАНИ МАТЕРИАЛЛАРИ ТЎПЛАМИ",
            "type": "(tezis 1) Termiz-2022"
        },
        {
            "url": "(tezis 2) Tashkent-2023.Mathematics, mechanics and intellectual technologies.pdf",
            "name": "ABSTRACTS OF THE II REPUBLICAN SCIENTIFIC AND PRACTICAL CONFERENCE OF YOUNG SCIENTISTS MATHEMATICS, MECHANICS AND INTELLECTUAL TECHNOLOGIES TASHKENT-2023",
            "type": "(tezis 2) Tashkent-2023.Mathematics"
        },
        {
            "url": "(tezis 3) Buxoro-24.05.2023_Fiz-mat va mex.dolzarb muam_xalqaro konf.pdf",
            "name": "АКТУАЛЬНЫЕ ПРОБЛЕМЫ ФИЗИКИ, МАТЕМАТИКИ И МЕХАНИКИ",
            "type": "(tezis 3) Buxoro-24.05.2023_Fiz-mat"
        },

        {
            "url": "(tezis 4) Rahmatullin_ÆÑº¿ßδ-ñ«¬½áñ«ó 27-28 may 2023.pdf",
            "name": "Издательство «Университет», 2023 Международная научно-практическая конференция «Рахматулинские чтения»",
            "type": "(tezis 4) Rahmatullin"
        },

        {
            "url": "(tezis 5)_VII éßÑ¼¿α¡«ú« è«¡úαÑßßá ¼áΓÑ¼áΓ¿¬«ó Γεα¬ß¬«ú« ¼¿αá_TWMS2023.pdf",
            "name": "TWO-DIMENSIONAL PROBLEM OF ANOMALOUS TRANSPORT IN A  TWO-ZONE FRACTAL POROUS MEDIUM",
            "type": "(tezis 5)_VII"
        },

        {
            "url": "(tezis 6) _Abstracts of Al-Khwarizmi 2023.pdf",
            "name": " ACTUAL PROBLEMS OF APPLIED  MATHEMATICS AND INFORMATION TECHNOLOGIES-AL-KHWARIZMI 2023",
            "type": "(tezis 6) _Abstracts of Al-Khwarizmi 2023"
        },

        {
            "url": "(tezis 7) _Qarshi_ 2024.pdf",
            "name": " AMALIY MATEMATIKANING ZAMONAVIY MUAMMOLARI VA ISTIQBOLLARI",
            "type": "(tezis 7) _Qarshi_ 2024"
        },

        {
            "url": "(tezis 8) _Termiz_ 2024.pdf",
            "name": "TA’LIM JARAYONIGA RAQAMLI TEXNOLOGIYALAR VA SUN’IY INTELLEKTNI JORIY ETISH ISTIQBOLLARI",
            "type": "(tezis 8) _Termiz_ 2024"
        },

        {
            "url": "ⁿ DGU 38241.pdf",
            "name": " Ikki zonali fraktal muhitda anomal modda ko’chishi jarayonini sonli tadqiq etish",
            "type": " DGU 38241"
        },

        {
            "url": "ⁿ DGU 38242.pdf",
            "name": " Adsorbsiya hodisasini hisobga olgan holda anomal modda ko`chishi jarayonlarini sonli modellashtirish",
            "type": " DGU 38242"
        },

        {
            "url": "(maqola) 2023 Mexanika muammolari.pdf",
            "name": " ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ ПРОЦЕССА АНОМАЛЬНОГО ПЕРЕНОСА ВЕЩЕСТВА ВО ФРАКТАЛЬНОЙ СРЕДЕ С ПОМОЩЬЮ ДРОБНОГО ДИФФЕРЕНЦИАЛЬНОГО УРАВНЕНИЯ ",
            "type": " (maqola) 2023 Mexanika muammolari"
        },

        {
            "url": "(maqola) 2023_Samdu axborotnoma.pdf",
            "name": " АНАЛИЗ ПЕРЕНОСА ВЕЩЕСТВА В ПОРИСТОЙ СРЕДЕ НА ОСНОВЕ ДИФФУЗИОННОГО УРАВНЕНИЯ С МНОГО-ЧЛЕННЫМИ ДРОБНЫМИ ПРОИЗВОДНЫМИ ПО ВРЕМЕНИ ",
            "type": " (maqola) 2023_Samdu axborotnoma"
        },

        {
            "url": "(scopus) AIP_Multiterm time fractional diffusion equation.pdf",
            "name": " Model of Solute Transport in a Porous Medium with Multi-Term Time Fractional Diffusion Equation",
            "type": " (scopus) AIP_Multiterm time fractional diffusion equation"
        },

        {
            "url": "(scopus) B.Khuzhayorov_A.Usmonov_F.B.Kholliev - Numerical Solution of Anomalous Solute in a Two-Zone Fractal Porous Medium.pdf",
            "name": "Numerical Solution of Anomalous Solute Transport in a Two-Zone Fractal Porous Medium",
            "type": " (scopus) B.Khuzhayorov_A.Usmonov_F.B.Kholliev - Numerical Solution of Anomalous Solute in a Two-Zone Fractal Porous Medium"
        },

        {
            "url": "(scopus) Computation_Anomalous Solute Transport Using Adsorption Effects and the.pdf",
            "name": "Anomalous Solute Transport Using Adsorption Effects and the Degradation of Solute",
            "type": " (scopus) Computation_Anomalous Solute Transport Using Adsorption Effects and the"
        },
    ]
    return render(request, 'published_works_list.html', {'works': works})
def api(request):
    return render(request, 'api.html')
def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            full_text += text + "\n"  

        return full_text
def extract_math_from_pdf(file_path):
    pdf_text = read_pdf(file_path)
    
    formula_pattern = r'\$\$?(.*?)\$\$?'  
    formulas = re.findall(formula_pattern, pdf_text, re.DOTALL)

    return formulas
def home(request):
    return render(request, "index.html")
def selects(request):
    return render(request, 'selects.html', {'items': items})
def get_item_description(request, item_id):
    item = next((item for item in items if item['id'] == item_id), None)
    if item:
        file_path = f'static/{item["file"]}'
        return render(request, 'item_description.html', {'file_path': file_path})
    return render(request, 'item_description.html', {'file_path': 'Item not found.'})   
def answer(request):
    item_id = request.GET.get('options')
    filename = 'answer.html'
    if item_id == '1':
        description = {
            "C0": {"name": "Объемная концентрация вещества на входе C0", "value": 0.1},
            "vm": {"name": "Осредненная скорость движения раствора", "value": 1e-4},
            "Dm": {"name": "Коэффициент диффузии", "value": 1e-5},
            "N": {"name": "Количество интервалов", "value": 100},
            "tau": {"name": "Шаг сетки по направлению", "value": 1},
            "h": {"name": "Шаг сетки по направлению", "value": 0.1},
            "tmax": {"name": "Максимальное время", "value": 900},
            "w": {"name": "Коэффициент переноса массы", "value": 1e-5},
            "tetim": {"name": "Пористость в immobile зоне", "value": 0.1},
            "tetm": {"name": "Пористость в mobile зоне", "value": 0.4},
            "gama": {"name": "Коэффициент переноса массы", "value": 0.6},
            "alpha": {"name": "Дробная производная по времени", "value": 0.9},
            "beta": {"name": "Дробная производная по координате", "value": 1.8}
        }
    elif item_id == '2':
        description = {
            "C0": {"name": "Объемные концентрации вещества", "value": 0.1},
            "vm": {"name": "Осредненная скорость движения раствора", "value": 1e-4},
            "Dm": {"name": "Коэффициент гидродинамической дисперсии", "value": 5e-5},
            "n": {"name": "Количество интервалов", "value": 20},
            "tau": {"name": "Шаг сетки по направлению", "value": 1},
            "h": {"name": "Шаг сетки по направлению", "value": 0.1},
            "tmax": {"name": "Максимальное время", "value": 900},
            "w": {"name": "Коэффициент переноса массы", "value": 1e-5},
            "tetim": {"name": "Пористость в immobile зоне", "value": 0.1},
            "tetm": {"name": "Пористость в mobile зоне", "value": 0.4},
            "gam": {"name": "Коэффициент переноса массы", "value": 0.6},
            "alpha": {"name": "Дробная производная по времени", "value": 0.9},
            "bet": {"name": "Дробная производная по координате", "value": 0.9},
            "kd": {"name": "Коэффициент адсорбции:", "value": 1e-4},
        }
        filename = 'answer1.html'
    elif item_id == '3':
        description = {
            "C0": {"name": "Объемные концентрации вещества", "value": 0.1},
            "Dm": {"name": "Коэффициент диффузии", "value": 1e-4},
            "N": {"name": "Количество интервалов", "value": 20},
            "tau": {"name": "Шаг сетки по направлению", "value": 0.5},
            "h": {"name": "Шаг сетки по направлению", "value": 0.01},
            "tmax": {"name": "Максимальное время", "value": 900},
            "gamma": {"name": "Дробная производная по координате", "value": 2},
            "alfa": {"name": "Дробная производная по времени", "value": 1},
            "beta1": {"name": "Дробная производная по времени:", "value": 0.7},
            "beta2": {"name": "Дробная производная по времени:", "value": 0.5},
            "r1": {"name": "Положительные константы:", "value": 1.5},
            "r2": {"name": "Положительные константы:", "value": 1},
        }
        filename = 'answer2.html'
    else:
        description = {}

    template = loader.get_template(filename)
    context = {'description': description, 'item_id': item_id}
    return HttpResponse(template.render(context, request))
@csrf_exempt
def result(request):
     if request.method == 'POST':
        options = int(request.POST.get('options'))
        if options == 1:
            C0 = float(request.POST.get('C0', 0.1))
            vm = float(request.POST.get('vm', 1e-4))
            Dm = float(request.POST.get('Dm', 1e-5))
            n = int(request.POST.get('n', 20))
            tau = float(request.POST.get('tau', 1))
            h = float(request.POST.get('h', 0.1))
            tmax = int(request.POST.get('tmax', 2700))
            w = float(request.POST.get('w', 1e-6))
            tetim = float(request.POST.get('tetim', 0.1))
            tetm = float(request.POST.get('tetm', 0.4))
            gam = float(request.POST.get('gam', 0.06))
            alpha = float(request.POST.get('alpha', 0.9))
            bet = float(request.POST.get('bet', 2))
            
            data = calculate_data(C0, vm, Dm, n, tau, h, tmax, w, tetim, tetm, gam, alpha, bet)
            return JsonResponse(data)
        elif options == 2:
            form = SimulationForm(request.POST)
            if form.is_valid():

                C0 = form.cleaned_data['C0']
                vm = form.cleaned_data['vm']
                Dm = form.cleaned_data['Dm']
                n = form.cleaned_data['n']
                tau = form.cleaned_data['tau']
                h = form.cleaned_data['h']
                tmax = form.cleaned_data['tmax']
                w = form.cleaned_data['w']
                tetim = form.cleaned_data['tetim']
                tetm = form.cleaned_data['tetm']
                gam = form.cleaned_data['gam']
                alpha = form.cleaned_data['alpha']
                bet = form.cleaned_data['bet']
                kd = form.cleaned_data['kd']
                f = 0.7
                pb = 1000
                mlim = 1e-6
                msim = 1e-8
                mlm = 1e-3
                msm = 1e-4

                Rm = (tetm + f * pb * kd)
                Rim = (tetim + (1 - f) * pb * kd)
                A1 = (tetm * mlm + f * pb * kd * msm)
                A2 = (tetim * mlim + (1 - f) * pb * kd * msim)
                F1 = 1 / ((gamma(3 - (bet + 1)) * (h**(bet + 1))) * Rm)

                # Boshlang`ich shartlar
                Cm = np.zeros((tmax + 1, n + 1))
                Cim = np.zeros((tmax + 1, n + 1))
                Cm2 = np.zeros(tmax + 1)
                Cm3 = np.zeros(tmax + 1)
                Cm4 = np.zeros(tmax + 1)
                Cim2 = np.zeros(tmax + 1)
                Cim3 = np.zeros(tmax + 1)
                Cim4 = np.zeros(tmax + 1)

                # Chegaraviy shartlar
                for k in range(tmax + 1):
                    Cm[k, 0] = C0

                # Asosiy qism
                for k in range(tmax):
                    for i in range(n + 1):
                        s1 = 0
                        for l1 in range(k):
                            if k > 0:
                                s1 += (Cim[l1 + 1, i] - Cim[l1, i]) * ((k - l1 + 1)**(1 - alpha) - (k - l1)**(1 - alpha))
                        Cim[k + 1, i] = Cim[k, i] - s1 + ((gamma(2 - alpha) * tau**alpha) / Rim) * ((w * (Cm[k, i] - Cim[k, i])) - A2 * Cim[k, i])
                    for i in range(1, n):
                        if (bet + 1) == 2:
                            s01 = Cm[k, i + 1] - 2 * Cm[k, i] + Cm[k, i - 1]
                        else:
                            s01 = 0
                            for l in range(i):
                                s01 += ((l + 1)**(2 - (bet + 1)) - (l)**(2 - (bet + 1))) * (Cm[k, i + 1 - l] - 2 * Cm[k, i - l] + Cm[k, i - 1 - l])
                        Cm[k + 1, i] = (tau * tetm * Dm * F1 * s01) - (tau * vm * tetm * (Cm[k, i + 1] - Cm[k, i - 1])) / (Rm * 2 * h) - ((tau * w) / Rm) * (Cm[k, i] - Cim[k, i]) - ((tau * A1) / Rm) * Cm[k, i] + Cm[k, i]
                    Cm[k + 1, n] = Cm[k + 1, n - 1]

                x = [round(i * h,1) for i in range(n + 1)]
                for k in range(tmax + 1):
                    Cm2[k] = Cm[k, 2]
                    Cm3[k] = Cm[k, 3]
                    Cm4[k] = Cm[k, 4]
                    Cim2[k] = Cim[k, 2]
                    Cim3[k] = Cim[k, 3]
                    Cim4[k] = Cim[k, 4]
                y = [k * tau for k in range(tmax + 1)]

                if settings.STATIC_ROOT:
                    static_dir = os.path.join(settings.STATIC_ROOT, 'data')
                else:
                    static_dir = os.path.join(settings.STATICFILES_DIRS[0], 'data')
                    os.makedirs(static_dir, exist_ok=True)
                Cm_path = os.path.join(static_dir, 'Cm.csv')
                Cim_path = os.path.join(static_dir, 'Cim.csv')
                Cm2_path = os.path.join(static_dir, 'C2.csv')
                Cm3_path = os.path.join(static_dir, 'C3.csv')
                Cm4_path = os.path.join(static_dir, 'C4.csv')

                with open(Cm_path, mode='w', newline='') as Cmfile:
                    writer = csv.writer(Cmfile, delimiter=';')
                    writer.writerow(list(Cm[tmax, :]))

                with open(Cim_path, mode='w', newline='') as Cimfile:
                    writer = csv.writer(Cimfile, delimiter=';')
                    writer.writerow(list(Cim[tmax, :]))

                with open(Cm2_path, mode='w', newline='') as Cm2file:
                    writer = csv.writer(Cm2file, delimiter=';')
                    writer.writerow(list(Cm2))

                with open(Cm3_path, mode='w', newline='') as Cm3file:
                    writer = csv.writer(Cm3file, delimiter=';')
                    writer.writerow(list(Cm3))

                with open(Cm4_path, mode='w', newline='') as Cm4file:
                    writer = csv.writer(Cm4file, delimiter=';')
                    writer.writerow(list(Cm4))
                return JsonResponse({
                    'x': x,
                    "y": y,
                    'Cm': Cm.tolist(),
                    'Cim': Cim.tolist(),
                    'Cm2': Cm2.tolist(),
                    'Cm3': Cm3.tolist(),
                    'Cm4': Cm4.tolist(),
                    'Cim2': Cim2.tolist(),
                    'Cim3': Cim3.tolist(),
                    'Cim4': Cim4.tolist(),
                    'tmax': tmax,
                    'tau': tau
                })
        elif options == 3:
            C0 = float(request.POST.get('C0', 3))
            Dm = float(request.POST.get('Dm', 0.1))
            n = int(request.POST.get('N', 20))
            tau = float(request.POST.get('tau', 0.5))
            h = float(request.POST.get('h', 0.01))
            tmax = int(request.POST.get('tmax', 2700))

            gam = float(request.POST.get('gamma', 2))
            alpha = float(request.POST.get('alfa', 1))
            betta1 = float(request.POST.get('beta1', 0.7))
            betta2 = float(request.POST.get('beta2', 0.5))
            r1 = float(request.POST.get('r1', 1.5))
            r2 = float(request.POST.get('r2', 1))
            T1 = 1 / (gamma(3 - gam) * h**gam)
            T2 = 1 / gamma(2 - alpha) * tau**alpha
            T3 = r1 / gamma(2 - betta1) * tau**betta1
            T4 = r2 / gamma(2 - betta2) * tau**betta2

            # Boshlang`ich shartlar
            C = np.zeros((tmax + 1, n + 1))
            C2 = np.zeros(tmax + 1)
            C3 = np.zeros(tmax + 1)
            C4 = np.zeros(tmax + 1)


            # Chegaraviy shartlar
            for k in range(tmax + 1):
                C[k, 0] = C0

            # Asosiy qism
            for k in range(tmax):
                for i in range(1, n):
                    s1 = 0
                    s2 = 0
                    s3 = 0
                    for l1 in range(k):
                        if k > 0:
                            s1 += (C[l1 + 1, i] - C[l1, i]) * ((k - l1 + 1)**(1 - alpha) - (k - l1)**(1 - alpha))
                            s2 += (C[l1 + 1, i] - C[l1, i]) * ((k - l1 + 1)**(1 - betta1) - (k - l1)**(1 - betta1))
                            s3 += (C[l1 + 1, i] - C[l1, i]) * ((k - l1 + 1)**(1 - betta2) - (k - l1)**(1 - betta2))
                    if gam == 2:
                        s01 = C[k, i + 1] - 2 * C[k, i] + C[k, i - 1]
                    else:
                        s01 = 0
                        for l in range(i):
                            s01 += (C[k, i - 1 - l] - 2 * C[k, i - l] + C[k, i + 1 - l]) * ((l + 1)**(2 - gam) - l**(2 - gam))
                    C[k + 1, i] = (Dm * T1 * s01 - T2 * (s1 - C[k, i]) - T3 * (s2 - C[k, i]) - T4 * (s3 - C[k, i])) / (T2 + T3 + T4)
                C[k + 1, n] = C[k + 1, n - 1]

            # Koordinata bo'yicha taqsimot

            x = [i * h for i in range(n + 1)]
            y = [k * tau for k in range(tmax + 1)]
            # Vaqt bo'yicha dinamikasi
            for k in range(tmax + 1):
                C2[k] = C[k, 2]
                C3[k] = C[k, 3]
                C4[k] = C[k, 4]

            if settings.STATIC_ROOT:
                static_dir = os.path.join(settings.STATIC_ROOT, 'data')
            else:
                static_dir = os.path.join(settings.STATICFILES_DIRS[0], 'data')
                os.makedirs(static_dir, exist_ok=True)
            C1_path = os.path.join(static_dir, 'C1.csv')
            C2_path = os.path.join(static_dir, 'C2.csv')
            C3_path = os.path.join(static_dir, 'C3.csv')
            # Natijaning oxirgi ustunini Excelga eksport qilish
            with open(C1_path, mode='w', newline='') as Cfile:
                writer = csv.writer(Cfile, delimiter=';')
                writer.writerow(list(C2))

            with open(C2_path, mode='w', newline='') as Cfile:
                writer = csv.writer(Cfile, delimiter=';')
                writer.writerow(list(C3))

            with open(C3_path, mode='w', newline='') as Cfile:
                writer = csv.writer(Cfile, delimiter=';')
                writer.writerow(list(C4))

            return JsonResponse({
                "C": C.tolist(),
                'C2': C2.tolist(),
                'C3': C3.tolist(),
                'C4': C4.tolist(),
                "x": x,
                "y": y,
                "tmax": tmax
            })
def export_csv(request, filename):
    Cm = np.array(request.GET.get('Cm').split(','), dtype=float)
    output = BytesIO()
    df = pd.DataFrame(Cm)
    df.to_csv(output, index=False, header=False)
    output.seek(0)
    response = HttpResponse(output, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename={filename}.csv'
    return response
def about(request): 
    return render(request, 'about.html')
def calculate_data(C0, vm, Dm, n, tau, h, tmax, w, tetim, tetm, gam, alpha, bet):
    tet = tetim / tetm
    A = 1 + (w * gamma(2 - alpha) * tau**alpha) / (gam * tetim)

    # Boshlang'ich shartlar
    Cm = np.zeros((tmax + 1, n + 1))
    Cim = np.zeros((tmax + 1, n + 1))
    Cm2 = np.zeros(tmax + 1)
    Cm3 = np.zeros(tmax + 1)
    Cm4 = np.zeros(tmax + 1)
    Cim2 = np.zeros(tmax + 1)
    Cim3 = np.zeros(tmax + 1)
    Cim4 = np.zeros(tmax + 1)

    # Chegaraviy shartlar
    for k in range(tmax + 1):
        Cm[k, 0] = C0

    # Asosiy qism
    for k in range(tmax):
        for i in range(n + 1):
            s1 = 0
            for l1 in range(k):
                if k > 0:
                    s1 += (Cim[l1 + 1, i] - Cim[l1, i]) * ((k - l1 + 1)**(1 - alpha) - (k - l1)**(1 - alpha))
            Cim[k + 1, i] = (Cim[k, i] - s1 + (gamma(2 - alpha) * tau**alpha * w) * Cm[k, i] / (gam * tetim)) / A

        for i in range(1, n):
            if bet == 2:
                s01 = Cm[k, i + 1] - 2 * Cm[k, i] + Cm[k, i - 1]
            else:
                s01 = 0
                for l in range(i):
                    s01 += ((l + 1)**(2 - bet) - (l)**(2 - bet)) * (Cm[k, i + 1 - l] - 2 * Cm[k, i - l] + Cm[k, i - 1 - l])
            Cm[k + 1, i] = (tau * Dm * s01) / (gamma(3 - bet) * (h**bet)) - tau * vm * (Cm[k, i] - Cm[k, i - 1]) / h - gam * tau * tetim * s1 / (tetm * gamma(2 - alpha) * tau**alpha) * (Cim[k + 1, i] - Cim[k, i]) + Cm[k, i]
        Cm[k + 1, n] = Cm[k + 1, n - 1]

    # Natijani JSON formatida qaytarish

    # Koordinata bo'yicha taqsimot
    x = [round(i * h,1) for i in range(n + 1)]
    
    # Vaqt bo'yicha dinamikasi
    y = [k * tau for k in range(tmax + 1)]
    
    for k in range(tmax + 1):
        Cm2[k] = Cm[k, 2]
        Cm3[k] = Cm[k, 3]
        Cm4[k] = Cm[k, 4]
        Cim2[k] = Cim[k, 2]
        Cim3[k] = Cim[k, 3]
        Cim4[k] = Cim[k, 4]
    if settings.STATIC_ROOT:
        static_dir = os.path.join(settings.STATIC_ROOT, 'data')
    else:
        static_dir = os.path.join(settings.STATICFILES_DIRS[0], 'data')
        os.makedirs(static_dir, exist_ok=True)

    Cm_path = os.path.join(static_dir, 'Cm.csv')
    Cim_path = os.path.join(static_dir, 'Cim.csv')
    Cm2_path = os.path.join(static_dir, 'C2.csv')
    Cm3_path = os.path.join(static_dir, 'C3.csv')
    Cm4_path = os.path.join(static_dir, 'C4.csv')

    with open(Cm_path, mode='w', newline='') as Cmfile:
        writer = csv.writer(Cmfile, delimiter=';')
        writer.writerow(list(Cm[tmax, :]))

    with open(Cim_path, mode='w', newline='') as Cimfile:
        writer = csv.writer(Cimfile, delimiter=';')
        writer.writerow(list(Cim[tmax, :]))

    with open(Cm2_path, mode='w', newline='') as Cm2file:
        writer = csv.writer(Cm2file, delimiter=';')
        writer.writerow(list(Cm2))

    with open(Cm3_path, mode='w', newline='') as Cm3file:
        writer = csv.writer(Cm3file, delimiter=';')
        writer.writerow(list(Cm3))

    with open(Cm4_path, mode='w', newline='') as Cm4file:
        writer = csv.writer(Cm4file, delimiter=';')
        writer.writerow(list(Cm4))
    return {
        'x': x,
        'Cm_tmax': Cm[tmax, :].tolist(),
        'Cm_2tmax3': Cm[2 * tmax // 3, :].tolist(),
        'Cm_tmax3': Cm[tmax // 3, :].tolist(),
        'Cim_tmax': Cim[tmax, :].tolist(),
        'Cim_2tmax3': Cim[2 * tmax // 3, :].tolist(),
        'Cim_tmax3': Cim[tmax // 3, :].tolist(),
        'y': y,
        'Cm2': Cm2.tolist(),
        'Cm3': Cm3.tolist(),
        'Cm4': Cm4.tolist(),
        'Cim2': Cim2.tolist(),
        'Cim3': Cim3.tolist(),
        'Cim4': Cim4.tolist(),
        'tmax': tmax,
    }
def calculate(request):
    if request.method == 'GET':
        C0 = float(request.GET.get('C0', 0.1))
        vm = float(request.GET.get('vm', 1e-4))
        Dm = float(request.GET.get('Dm', 1e-5))
        n = int(request.GET.get('n', 20))
        tau = float(request.GET.get('tau', 1))
        h = float(request.GET.get('h', 0.1))
        tmax = int(request.GET.get('tmax', 2700))
        w = float(request.GET.get('w', 1e-6))
        tetim = float(request.GET.get('tetim', 0.1))
        tetm = float(request.GET.get('tetm', 0.4))
        gam = float(request.GET.get('gam', 0.06))
        alpha = float(request.GET.get('alpha', 0.9))
        bet = float(request.GET.get('bet', 2))
        
        data = calculate_data(C0, vm, Dm, n, tau, h, tmax, w, tetim, tetm, gam, alpha, bet)
        return JsonResponse(data)

    return render(request, 'index.html')