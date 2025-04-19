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
LOGGER = Logger('SUL DATA HANDLER')
DATA_INIZIO_FILTRO = config['GR3N']['DATA_INIZIO_FILTRO']
DATA_FINE_FILTRO = config['GR3N']['DATA_FINE_FILTRO']

def is_chg_pt(curr, prev):
    return  (curr[0] > COPPIA_MIDPOINT and prev[0] < COPPIA_MIDPOINT and curr[1] > prev[1]) or \
            (curr[0] > COPPIA_MIDPOINT and prev[0] < COPPIA_MIDPOINT and curr[1] < prev[1]) or \
            (curr[0] < COPPIA_MIDPOINT and prev[0] > COPPIA_MIDPOINT and curr[1] > prev[1]) or \
            (curr[0] < COPPIA_MIDPOINT and prev[0] > COPPIA_MIDPOINT and curr[1] < prev[1])

def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    coppia_sig = signals[1]
    coppia = {pt.timestamp: (i, pt.value) for i, pt in enumerate(coppia_sig.points)}

    curr_coppia_index, curr_coppia = coppia[t]
    if curr_coppia_index > 0:
        try:
            prev_index = [tup[0] for tup in coppia.values() if tup[0] < curr_coppia_index][-1]
            prev_coppia = coppia_sig.points[prev_index].value
        except IndexError:
            prev_coppia = None
    else:
        prev_coppia = curr_coppia

    cicloattivo_sig = signals[2]
    cicloattivo = {pt.timestamp: (i, pt.value) for i, pt in enumerate(cicloattivo_sig.points)}

    curr_cicloattivo_index, curr_cicloattivo = cicloattivo[t]
    if curr_cicloattivo_index > 0:
        try:
            prev_index = [tup[0] for tup in cicloattivo.values() if tup[0] < curr_cicloattivo_index][-1]
            prev_cicloattivo = cicloattivo_sig.points[prev_index].value
        except IndexError:
            prev_cicloattivo = None
    else:
        prev_cicloattivo = curr_cicloattivo

    identified_event = None
    if (curr_coppia > COPPIA_MIDPOINT and prev_coppia < COPPIA_MIDPOINT and curr_cicloattivo > prev_cicloattivo):
        identified_event = events[0]
    elif (curr_coppia > COPPIA_MIDPOINT and prev_coppia < COPPIA_MIDPOINT and curr_cicloattivo < prev_cicloattivo):
        identified_event = events[1]
    elif (curr_coppia < COPPIA_MIDPOINT and prev_coppia > COPPIA_MIDPOINT and curr_cicloattivo > prev_cicloattivo):
        identified_event = events[2]
    elif (curr_coppia < COPPIA_MIDPOINT and prev_coppia > COPPIA_MIDPOINT and curr_cicloattivo < prev_cicloattivo):
        identified_event = events[3]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: datetime):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def parse_data(path: str):
    cicloattivo: SampledSignal = SampledSignal([], label='ca')
    assorbimento: SampledSignal = SampledSignal([], label='a')
    coppia: SampledSignal = SampledSignal([], label='cp')

    dd_real = pd.read_csv(path)

    # First, convert to datetime so that I can easily sort them
    dd_cicloattivo = dd_real[dd_real['DataObjectField'] == 'CicloAttivo']
    dd_cicloattivo.loc[:, 'time'] = pd.to_datetime(dd_cicloattivo['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_cicloattivo.sort_values(by='time')

    dd_assorbimento = dd_real[dd_real['DataObjectField'] == 'Assorbimento']
    dd_assorbimento.loc[:, 'time'] = pd.to_datetime(dd_assorbimento['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_assorbimento.sort_values(by='time')

    dd_coppia = dd_real[dd_real['DataObjectField'] == 'Coppia']
    dd_coppia.loc[:, 'time'] = pd.to_datetime(dd_coppia['time'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_coppia.sort_values(by='time')

    # Then, back to Timestamp so that doesn't make conflicts with the code
    cicloattivo.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_cicloattivo.iterrows()])
    assorbimento.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_assorbimento.iterrows()])
    coppia.points.extend([SignalPoint(parse_ts(record['time']), record['Value']) for index, record in dd_coppia.iterrows()])

    return [assorbimento, coppia, cicloattivo]


def get_absorption_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_abs = sum([pt.value for pt in segment])
    avg_abs = sum_abs / (len(segment))
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

    for timestamp in trace.t:
        dt = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.min,
            timestamp.sec)
        ax.plot([dt], [dd_assorbimento['Value'].max()*1.10], 'rv')
        ax.vlines(x=dt, ymin=0, ymax=dd_assorbimento['Value'].max()*1.10, color='r', linestyle=':', alpha=0.5)

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

def plot_coppia_eventi(path: str, filename: str, trace: TimedTrace):
    dd_real = pd.read_csv(path)
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
    plt.xticks(rotation=45)

    plt.title(filename)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()