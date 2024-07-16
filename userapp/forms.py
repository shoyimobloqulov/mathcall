from django import forms

class CalculationForm(forms.Form):
    C0 = forms.FloatField(label='Объемная концентрация вещества на входе C0', initial=0.1)
    vm = forms.FloatField(label='Осредненная скорость движения раствора', initial=1e-4)
    Dm = forms.FloatField(label='Коэффициент диффузии', initial=1e-5)
    N = forms.IntegerField(label='Количество интервалов', initial=100)
    tau = forms.FloatField(label='Шаг сетки по направлению 1', initial=1)
    h = forms.FloatField(label='Шаг сетки по направлению 2', initial=0.1)
    tmax = forms.IntegerField(label='Максимальное время', initial=3600)
    w = forms.FloatField(label='Коэффициент переноса массы', initial=1e-5)
    tetim = forms.FloatField(label='Пористость в immobile зоне', initial=0.1)
    tetm = forms.FloatField(label='Пористость в mobile зоне', initial=0.4)
    gama = forms.FloatField(label='Коэффициент переноса массы', initial=0.6)
    alpha = forms.FloatField(label='Дробная производная по времени', initial=0.9)
    beta = forms.FloatField(label='Дробная производная по координате', initial=1.8)
