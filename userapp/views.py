from django.shortcuts import render # type: ignore
from django.http import HttpResponse # type: ignore
from .utils import read_pdf_file
import pdfplumber # type: ignore
import re
from django.template import loader # type: ignore

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.special import gamma # type: ignore

from django.shortcuts import render # type: ignore
from django.http import HttpResponse # type: ignore
from .forms import CalculationForm
import csv
from io import BytesIO
import base64
from .tasks import perform_calculation
from django.http import JsonResponse # type: ignore
from scipy.special import gamma # type: ignore
import json

import csv
import pandas as pd

items = [
    {'id': '1', 'name': 'Аномальный перенос вещества в двухзонной фрактальной среде', 'file': 'pdf_files/Аномальный перенос вещества в двухзонной фрактальной среде.pdf'},
    {'id': '2', 'name': 'Аномальный перенос с учетом адсорбционных эффектов и разложения вещества', 'file': 'pdf_files/Аномальный перенос с учетом адсорбционных эффектов и разложения вещества.pdf'},
    {'id': '3', 'name': 'Аномальный перенос с много – членными дробными производными', 'file': 'pdf_files/Аномальный перенос с много – членными дробными производными.pdf'},
]

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
        description = read_pdf_file(file_path)
        return render(request, 'item_description.html', {'file_path': file_path})
    return render(request, 'item_description.html', {'file_path': 'Item not found.'})

def calculate(request):
    if request.method == 'GET':
        return render(request, 'diogram.html', {'data': request.GET})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
def answer(request):
    item_id = request.GET.get('options')

    if item_id == '1':
        description = {
            "C0": {"name": "Объемная концентрация вещества на входе C0", "value": 0.1},
            "vm": {"name": "Осредненная скорость движения раствора", "value": 1e-4},
            "Dm": {"name": "Коэффициент диффузии", "value": 1e-5},
            "N": {"name": "Количество интервалов", "value": 100},
            "tau": {"name": "Шаг сетки по направлению", "value": 1},
            "h": {"name": "Шаг сетки по направлению", "value": 0.1},
            "tmax": {"name": "Максимальное время", "value": 3600},
            "w": {"name": "Коэффициент переноса массы", "value": 1e-5},
            "tetim": {"name": "Пористость в immobile зоне", "value": 0.1},
            "tetm": {"name": "Пористость в mobile зоне", "value": 0.4},
            "gama": {"name": "Коэффициент переноса массы", "value": 0.6},
            "alpha": {"name": "Дробная производная по времени", "value": 0.9},
            "beta": {"name": "Дробная производная по координате", "value": 1.8}
        }
    elif item_id == '2':
        description = {
            "C0": {"name": "Объемные концентрации вещества", "value": 0.2},
            "vm": {"name": "Осредненная скорость движения раствора", "value": 2e-4},
            "Dm": {"name": "Коэффициент гидродинамической дисперсии", "value": 2e-5},
            "N": {"name": "Количество интервалов", "value": 150},
            "dx1": {"name": "Шаг сетки по направлению", "value": 0.2},
            "dx2": {"name": "Шаг сетки по направлению", "value": 0.2},
            "tmax": {"name": "Максимальное время", "value": 7200},
            "w": {"name": "Коэффициент переноса массы", "value": 2e-5},
            "tetim": {"name": "Пористость в immobile зоне", "value": 0.2},
            "tetm": {"name": "Пористость в mobile зоне", "value": 0.5},
            "gam": {"name": "Коэффициент переноса массы", "value": 0.7},
            "alpha": {"name": "Дробная производная по времени", "value": 0.8},
            "bet": {"name": "Дробная производная по координате", "value": 1.7},
            "adsorption_centers": {"name": "Представляют доли центров адсорбции", "value": [0.25, 0.35]},
            "porous_density": {"name": "Объемная плотность пористой среды", "value": 0.3},
            "adsorption_coefficient": {"name": "Коэффициент адсорбции", "value": 0.01},
            "first_order_decomposition_mobile": {"name": "Коэффициенты разложения первого порядка для разложения растворенного вещества в областях с подвижной и неподвижной жидкостью", "value": [0.001, 0.002]},
            "first_order_decomposition_immobile": {"name": "Коэффициенты разложения вещества первого порядка в подвижной и неподвижной адсорбированных твердых фазах", "value": [0.003, 0.004]}
        }
    elif item_id == '3':
        description = {
            "C0": {"name": "Объемные концентрации вещества", "value": 0.3},
            "Dm": {"name": "Коэффициент диффузии", "value": 3e-5},
            "N": {"name": "Количество интервалов", "value": 200},
            "dx1": {"name": "Шаг сетки по направлению", "value": 0.3},
            "dx2": {"name": "Шаг сетки по направлению", "value": 0.3},
            "tmax": {"name": "Максимальное время", "value": 10800},
            "alpha": {"name": "Дробная производная по времени", "value": 0.7},
            "bet": {"name": "Дробная производная по координате", "value": 1.6},
            "gamma": {"name": "Коэффициент переноса массы", "value": 0.5},
            "delta": {"name": "Коэффициент переноса массы", "value": 0.4},
            "positive_constants": {"name": "Положительные константы", "value": [0.5, 0.6]}
        }
    else:
        description = {}
    template = loader.get_template('answer.html')
    context = {'description': description, 'item_id': item_id}
    return HttpResponse(template.render(context, request))
def calculate_and_plot(request):
    data = json.dumps(request.GET)
    return render(request, "calculate_and_plot.html", {"data": data})

def export_csv(request, filename):
    Cm = np.array(request.GET.get('Cm').split(','), dtype=float)
    output = BytesIO()
    df = pd.DataFrame(Cm)
    df.to_csv(output, index=False, header=False)
    output.seek(0)
    response = HttpResponse(output, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename={filename}.csv'
    return response

def functiondata(request,item_id):
    print(request,item_id)
    if int(item_id) == 1:
        C0 = 0.1
        vm = 1e-4
        Dm = 1e-5
        n = 20
        tau = 1
        h = 0.1
        tmax = 2700
        w = 1e-6
        tetim = 0.1
        tetm = 0.4
        gam = 0.06
        alpha = 0.9
        bet = 2
        return datafunctionone(C0,vm,Dm,n,tau,h,tmax,w,tetim,tetm,gam,alpha,bet)
    
def about(request): 
    return render(request, 'about.html')

def datafunctionone(C0,vm,Dm,n,tau,h,tmax,w,tetim,tetm,gam,alpha,bet):
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

    data = {
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


    return JsonResponse(data)