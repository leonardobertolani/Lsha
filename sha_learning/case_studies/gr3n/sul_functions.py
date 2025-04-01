import configparser
import pandas as pd
import os
from typing import List, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sha_learning.domain.lshafeatures import Event, FlowCondition, TimedTrace
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

try:
    CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
except ValueError:
    CS_VERSION = None

COPPIA_MIDPOINT = int(config['GR3N']['COPPIA_MIDPOINT'])
DIFF_LOW = int(config['GR3N']['DIFF_LOW'])
DIFF_HIGH = int(config['GR3N']['DIFF_HIGH'])
LOGGER = Logger('SUL DATA HANDLER')
DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']

def is_chg_pt(curr, prev):
    return  (curr[0] > COPPIA_MIDPOINT and prev[0] < COPPIA_MIDPOINT) or \
            (curr[0] < COPPIA_MIDPOINT and prev[0] > COPPIA_MIDPOINT) or \
            (curr[0] < DIFF_LOW and prev[0] > DIFF_HIGH) or \
            (curr[0] > DIFF_HIGH and prev[0] < DIFF_LOW) or \
            (curr[0] < DIFF_HIGH and prev[0] < DIFF_LOW) or \
            (curr[0] < DIFF_LOW and prev[0] < DIFF_HIGH) or \
            (curr[0] > DIFF_HIGH and prev[0] > DIFF_LOW) or \
            (curr[0] > DIFF_LOW and prev[0] > DIFF_HIGH)

def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    coppia_sig = signals[1]
    coppia = {pt.timestamp: (i, pt.value) for i, pt in enumerate(coppia_sig.points)}

    diff_sig = signals[2]
    diff = {pt.timestamp: (i, pt.value) for i, pt in enumerate(diff_sig.points)}

    curr_coppia_index, curr_coppia = coppia[t]
    if curr_coppia_index > 0:
        try:
            prev_index = [tup[0] for tup in coppia.values() if tup[0] < curr_coppia_index][-1]
            prev_coppia = coppia_sig.points[prev_index].value
        except IndexError:
            prev_coppia = None
    else:
        prev_coppia = curr_coppia

    curr_diff_index, curr_diff = diff[t]
    if curr_diff_index > 0:
        try:
            prev_index = [tup[0] for tup in diff.values() if tup[0] < curr_diff_index][-1]
            prev_diff = diff_sig.points[prev_index].value
        except IndexError:
            prev_diff = None
    else:
        prev_diff = curr_diff

    identified_event = None
    if (curr_coppia > COPPIA_MIDPOINT and prev_coppia < COPPIA_MIDPOINT):
        identified_event = events[0]
    elif (curr_coppia < COPPIA_MIDPOINT and prev_coppia > COPPIA_MIDPOINT):
        identified_event = events[1]


    if (curr_diff > DIFF_HIGH and prev_diff < DIFF_LOW):
        identified_event = events[6]
    elif (curr_diff < DIFF_LOW and prev_diff > DIFF_HIGH):
        identified_event = events[7]
    elif (curr_diff > DIFF_LOW and prev_diff < DIFF_LOW):
        identified_event = events[3]
    elif (curr_diff < DIFF_LOW and prev_diff > DIFF_LOW):
        identified_event = events[2]
    elif (curr_diff > DIFF_LOW and prev_diff > DIFF_HIGH):
        identified_event = events[5]
    elif (curr_diff > DIFF_HIGH and prev_diff > DIFF_LOW):
        identified_event = events[4]


    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: datetime):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def parse_data(path: str):
    differenziale: SampledSignal = SampledSignal([], label='df')
    assorbimento: SampledSignal = SampledSignal([], label='a')
    coppia: SampledSignal = SampledSignal([], label='cp')

    dd_real = pd.read_csv('D:\\Uni\\Magistrale\\1 Anno\\1 semestre\\Software engineering 2\\Gr3n\\csv\\20250202_DecanterData_REAL.csv')

    dd_differenziale = dd_real[dd_real['DataObjectField'] == 'Differenziale']
    dd_differenziale.loc[:, 'time'] = pd.to_datetime(dd_differenziale['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_differenziale.sort_values(by='time')

    dd_assorbimento = dd_real[dd_real['DataObjectField'] == 'Assorbimento']
    dd_assorbimento.loc[:, 'time'] = pd.to_datetime(dd_assorbimento['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_assorbimento.sort_values(by='time')

    dd_coppia = dd_real[dd_real['DataObjectField'] == 'Coppia']
    dd_coppia.loc[:, 'time'] = pd.to_datetime(dd_coppia['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_coppia.sort_values(by='time')

    data_inizio_filtraggio = pd.to_datetime(DATA_INIZIO_FILTRO)
    data_fine_filtraggio = pd.to_datetime(DATA_FINE_FILTRO)

    dd_differenziale_dettaglio = dd_differenziale[(dd_differenziale['time'] >= data_inizio_filtraggio) & (dd_differenziale['time'] <= data_fine_filtraggio)]
    dd_assorbimento_dettaglio = dd_assorbimento[(dd_assorbimento['time'] >= data_inizio_filtraggio) & (dd_assorbimento['time'] <= data_fine_filtraggio)]
    dd_coppia_dettaglio = dd_coppia[(dd_coppia['time'] >= data_inizio_filtraggio) & (dd_coppia['time'] <= data_fine_filtraggio)]

    differenziale.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_differenziale_dettaglio.iterrows()])
    assorbimento.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_assorbimento_dettaglio.iterrows()])
    coppia.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_coppia_dettaglio.iterrows()])

    return [assorbimento, coppia, differenziale]


def get_absorption_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_abs = sum([pt.value for pt in segment])
    avg_abs = sum_abs / len(segment)
    return avg_abs


def plot_assorbimento_eventi(trace: TimedTrace):
    dd_real = pd.read_csv(
        'D:\\Uni\\Magistrale\\1 Anno\\1 semestre\\Software engineering 2\\Gr3n\\csv\\20250202_DecanterData_REAL.csv')
    dd_assorbimento = dd_real[dd_real['DataObjectField'] == 'Assorbimento']
    dd_assorbimento.loc[:, 'time'] = pd.to_datetime(dd_assorbimento['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_assorbimento = dd_assorbimento.sort_values(by='time')

    fig, ax = plt.subplots(figsize=(12, 6))

    xaxis_assorbimento = [record['time'] for index, record in dd_assorbimento.iterrows()]
    yaxis_assorbimento = [record['Value'] for index, record in dd_assorbimento.iterrows()]
    ax.plot(xaxis_assorbimento, yaxis_assorbimento, label='Assorbimento')

    for index in range(0, len(trace.t)):
        timestamp = trace.t[index]
        event = trace.e[index]
        dt = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.min,
            timestamp.sec)
        ax.plot([dt], [dd_assorbimento['Value'].max()*1.10], 'rv')

        if(event.symbol.startswith('cp')):
            ax.vlines(x=dt, ymin=0, ymax=dd_assorbimento['Value'].max()*1.10, color='r', linestyle=':', alpha=0.5)
        else:
            ax.vlines(x=dt, ymin=0, ymax=dd_assorbimento['Value'].max()*1.10, color='y', linestyle=':', alpha=0.5)

    # Formattazione dell'asse delle date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
    DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']
    ax.set_xlim(pd.to_datetime(DATA_INIZIO_FILTRO), pd.to_datetime(DATA_FINE_FILTRO))
    plt.xticks(rotation=45)

    plt.title('Assorbimento con Eventi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

def plot_coppia_eventi(trace: TimedTrace):
    dd_real = pd.read_csv(
        'D:\\Uni\\Magistrale\\1 Anno\\1 semestre\\Software engineering 2\\Gr3n\\csv\\20250202_DecanterData_REAL.csv')
    dd_coppia = dd_real[dd_real['DataObjectField'] == 'Coppia']
    dd_coppia.loc[:, 'time'] = pd.to_datetime(dd_coppia['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_coppia = dd_coppia.sort_values(by='time')

    fig, ax = plt.subplots(figsize=(12, 6))

    xaxis_assorbimento = [record['time'] for index, record in dd_coppia.iterrows()]
    yaxis_assorbimento = [record['Value'] for index, record in dd_coppia.iterrows()]
    ax.plot(xaxis_assorbimento, yaxis_assorbimento, label='Coppia')#, marker='o')

    for timestamp in trace.t:
        dt = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.min,
            timestamp.sec)
        ax.plot([dt], [dd_coppia['Value'].max()*1.10], 'rv')
        ax.vlines(x=dt, ymin=0, ymax=dd_coppia['Value'].max()*1.10, color='r', linestyle=':', alpha=0.5)

    # Formattazione dell'asse delle date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
    DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']
    ax.set_xlim(pd.to_datetime(DATA_INIZIO_FILTRO), pd.to_datetime(DATA_FINE_FILTRO))
    plt.xticks(rotation=45)

    plt.title('Coppia con Eventi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

def plot_diff_eventi(trace: TimedTrace):
    dd_real = pd.read_csv(
        'D:\\Uni\\Magistrale\\1 Anno\\1 semestre\\Software engineering 2\\Gr3n\\csv\\20250202_DecanterData_REAL.csv')
    dd_diff = dd_real[dd_real['DataObjectField'] == 'Differenziale']
    dd_diff.loc[:, 'time'] = pd.to_datetime(dd_diff['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_diff = dd_diff.sort_values(by='time')

    fig, ax = plt.subplots(figsize=(12, 6))

    xaxis_diff = [record['time'] for index, record in dd_diff.iterrows()]
    yaxis_diff = [record['Value'] for index, record in dd_diff.iterrows()]
    ax.plot(xaxis_diff, yaxis_diff, label='Differenziale')#, marker='o')

    for timestamp in trace.t:
        dt = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.min,
            timestamp.sec)
        ax.plot([dt], [dd_diff['Value'].max()*1.10], 'rv')
        ax.vlines(x=dt, ymin=0, ymax=dd_diff['Value'].max()*1.10, color='r', linestyle=':', alpha=0.5)

    # Formattazione dell'asse delle date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
    DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']
    ax.set_xlim(pd.to_datetime(DATA_INIZIO_FILTRO), pd.to_datetime(DATA_FINE_FILTRO))
    plt.xticks(rotation=45)

    plt.title('Differenziale con Eventi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()