import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from reaching_contact_audit import first_event


def test_first_event_ignores_pre_push_and_separates_transient_from_sustained():
    times=np.arange(8)*.01
    mask=np.array([True,True,False,True,False,True,True,True])
    assert first_event(times,mask,.02)==.03
    assert first_event(times,mask,.02,.02)==.05
    assert first_event(times,mask,.02,.03) is None


def test_first_event_no_crossing():
    assert first_event(np.arange(5),np.zeros(5,dtype=bool),0) is None
