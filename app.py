import pandas as pd
import streamlit as st
from bokeh.plotting import figure
from bokeh.transform import cumsum
import numpy as np
from math import pi
from scipy.stats import ttest_ind
from scipy.stats import chisquare

st.header("Домашнее задание по темам: 'Статистика', 'Визуализация', 'Развертывание в виде веб-приложения'")

uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

def create_pie_chart(dataframe, column_name):
    value_counts = dataframe[column_name].value_counts()
    pie_data = pd.DataFrame({'values': value_counts.values}, index=value_counts.index.to_list())
    pie_data['angle'] = pie_data['values']/pie_data['values'].sum() * 2*pi
    num_colors = len(pie_data)
    colors = np.random.rand(num_colors, 3)
    
    pie_data['colors'] = [f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for r, g, b in colors]

    p = figure(plot_height=400, title=f'Распределение {column_name}', toolbar_location=None,
               tools="hover", tooltips="@index: @values", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    line_color="white", fill_color='colors', legend_field='index', source=pie_data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    st.bokeh_chart(p, use_container_width=True)

def create_histogram(dataframe, column_name):
    st.header(f"Гистограмма и плотность вероятности для переменной: {column_name}")

    hist_data = dataframe[column_name]
    
    bins = np.linspace(hist_data.min(), hist_data.max(), 40)
    hist, edges = np.histogram(hist_data, density=True, bins=bins)
    
    x = np.linspace(hist_data.min(), hist_data.max(), 100)
    pdf = np.exp(-0.5 * ((x - hist_data.mean()) / hist_data.std())**2) / (hist_data.std() * np.sqrt(2.0*np.pi))
    
    p = figure(plot_height=400, title=f'Гистограмма и плотность вероятности {column_name}',
                toolbar_location="right",
                x_range=(hist_data.min(), hist_data.max()))

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="skyblue", line_color="white", legend_label="Гистограмма")

    p.line(x, pdf, line_width=2, line_color="navy", legend_label="Плотность вероятности")

    p.y_range.start = 0
    p.xaxis.axis_label = column_name
    p.yaxis.axis_label = "PDF({})".format(column_name)
    p.legend.title = "Легенда"
    p.grid.grid_line_color="white"

    st.bokeh_chart(p, use_container_width=True)

def ttest(dataframe, first_column, second_column):
    st.header("Независимые выборки T-Test")
    
    first_column = dataframe[first_column]
    second_column = dataframe[second_column]
    
    t_statistic, p_value = ttest_ind(first_column, second_column)
    
    st.write(f"T-Statistic: {t_statistic}")
    st.write(f"P-Value: {p_value}")
    
    alpha = 0.05
    if p_value < alpha:
        st.write("Отклонение нулевой гипотезы: Между группами существует значимая разница.")
    else:
        st.write("Не удается отвергнуть нулевую гипотезу: Значимых различий между группами нет.")

def chi_square_test(dataframe, first_variable, second_variable):
    st.header("Тест Хи-квадрат (Chi-Square Test)")

    contingency_table = pd.crosstab(dataframe[first_variable], dataframe[second_variable])
    chi2, p_value = chisquare(contingency_table.values) 

    st.write(f"Chi-Square Statistic: {chi2}")
    st.write(f"P-Value: {p_value[0]}")  

    alpha = 0.05
    if p_value[0] < alpha:
        st.write("Отклонение нулевой гипотезы: Между переменными существует значимая связь.")
    else:
        st.write("Не удается отвергнуть нулевую гипотезу: Связи между переменными нет.")

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

    first_variable = st.selectbox("Выберите первую переменную для исследования", dataframe.columns)
    first_unique_values = dataframe[first_variable].unique()
    
    if dataframe[first_variable].dtype in ['object']:
        create_pie_chart(dataframe, first_variable)
    elif len(first_unique_values) < 2 or all(val in [0, 1, 2, 3] for val in first_unique_values):
        create_pie_chart(dataframe, first_variable)
    elif dataframe[first_variable].dtype in ['int64', 'float']:
        create_histogram(dataframe, first_variable)
    else:
        st.write('Данный тип данных неизвестен.')

    second_variable = st.selectbox("Выберите вторую переменную для исследования", dataframe.columns)
    second_unique_values = dataframe[second_variable].unique()

    if dataframe[second_variable].dtype in ['object']:
        create_pie_chart(dataframe, second_variable)
    elif len(second_unique_values) < 2 or all(val in [0, 1, 2, 3] for val in second_unique_values):
        create_pie_chart(dataframe, second_variable)    
    elif dataframe[second_variable].dtype in ['int64', 'float']:
        create_histogram(dataframe, second_variable)
    else:
        st.write('Данный тип данных неизвестен.')
    
    algorithm = st.selectbox("Выберите алгоритм для проверки гипотез. 't-test' используйте для некатегориальных данных; 'chi-square test' -- используйте для категориальных данных", ('t-test', 'chi-square test'))

    if algorithm == 't-test':

        st.write('Вы выбрали:', algorithm)

        if dataframe[first_variable].dtype in ['int64', 'float'] and dataframe[second_variable].dtype in ['int64', 'float']:
            ttest(dataframe, first_variable, second_variable)
        else:
            st.write('Пожалуйста, выберите столбцы содержащие некатегориальные типы данных.')

    elif algorithm == 'chi-square test':

        st.write('Вы выбрали:', algorithm)

        if ((dataframe[first_variable].dtype == 'object' or len(first_unique_values) < 2 or all(val in [0, 1, 2, 3] for val in first_unique_values)) and
            (dataframe[second_variable].dtype == 'object' or len(second_unique_values) < 2 or all(val in [0, 1, 2, 3] for val in second_unique_values))):
            chi_square_test(dataframe, first_variable, second_variable)
        else:
            st.write('Пожалуйста, выберите столбцы содержащие категориальные типы данных.')

    
    