import configparser
from typing import List

import src.ekg_extractor.mgrs.db_connector as conn
from it.polimi.hri_learn.case_studies.auto_twin.sul_functions import label_event, parse_data, get_rand_param, \
    is_chg_pt
from it.polimi.hri_learn.domain.lshafeatures import Event, ProbDistribution
from it.polimi.hri_learn.domain.sigfeatures import Timestamp
from it.polimi.hri_learn.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition
from src.ekg_extractor.mgrs.ekg_queries import Ekg_Querier
from src.ekg_extractor.model.semantics import EntityForest
from it.polimi.hri_learn.lstar_sha.teacher import Teacher

config = configparser.ConfigParser()
config.sections()
config.read('./resources/config/config.ini')
config.sections()


def foo_model(interval: List[Timestamp]):
    return interval


# define flow conditions
foo_fc: FlowCondition = FlowCondition(0, foo_model)

# define distributions
foo_distr = ProbDistribution(0, {'avg': 0.0})

model2distr = {0: []}
s_id = RealValuedVar([foo_fc], [], model2distr, label='s_id')

# define events
driver = conn.get_driver()
querier: Ekg_Querier = Ekg_Querier(driver)
unique_events = querier.get_unique_events()
events: List[Event] = [Event('', e.replace('Pass Sensor ', ''), e.replace('Pass Sensor ', '').lower()) for e in
                       unique_events]

DRIVER_SIG = ['s_id']
DEFAULT_M = 0
DEFAULT_DISTR = 0

args = {'name': 'auto_twin', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}
auto_twin_cs = SystemUnderLearning([s_id], events, parse_data, label_event, get_rand_param, is_chg_pt, args=args)

conn.close_connection(driver)
test = False
if test:
    driver = conn.get_driver()
    querier: Ekg_Querier = Ekg_Querier(driver)

    TEST_N = 30

    labels_hierarchy = querier.get_entity_labels_hierarchy()

    evt_seqs = []
    if config['AUTO-TWIN CONFIGURATION']['POV'].lower() == 'item':
        entities = querier.get_items(labels_hierarchy=labels_hierarchy, limit=TEST_N, random=True)
    else:
        entities = querier.get_resources(labels_hierarchy=labels_hierarchy, limit=TEST_N, random=True)

    START_T = int(config['AUTO-TWIN CONFIGURATION']['START_T'])
    END_T = int(config['AUTO-TWIN CONFIGURATION']['END_T'])

    for entity in entities[:TEST_N]:
        entity_tree = querier.get_entity_tree(entity.entity_id, EntityForest([]), reverse=True)
        pov = config['AUTO-TWIN CONFIGURATION']['POV'].lower()
        if pov == 'item':
            events = querier.get_events_by_entity_tree(entity_tree[0], pov)
        else:
            events = querier.get_events_by_entity_tree_and_timestamp(entity_tree[0], START_T, END_T, pov)
        if len(events) > 0:
            evt_seqs.append(events)

    teacher = Teacher(auto_twin_cs)

    for seq in evt_seqs:
        auto_twin_cs.process_data(seq)
        print(auto_twin_cs.traces[-1])
        # auto_twin_cs.plot_trace(-1)
        id_cluster = teacher.ht_query(auto_twin_cs.traces[-1], foo_fc)
        print(id_cluster)

    conn.close_connection(driver)
