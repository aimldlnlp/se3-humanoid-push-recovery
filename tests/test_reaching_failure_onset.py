import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_failure_onset import event_start


def test_onset_uses_run_start_and_does_not_join_interrupted_events():
    t=np.arange(8)*.01
    mask=np.array([False,True,True,False,True,True,True,True])
    assert event_start(t,mask)==1
    assert event_start(t,mask,.025)==4
    assert event_start(t,mask,.04)==-1
    assert event_start(t,np.zeros(8,dtype=bool))==-1
