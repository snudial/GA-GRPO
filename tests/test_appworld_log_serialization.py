from datetime import datetime
import json
from types import SimpleNamespace

from fastapi.encoders import jsonable_encoder
from appworld_ga.log_serialization import diagnostic_json, save_request_log


def test_normal_api_log_preserves_official_encoding():
    value={'method':'post','url':'http://spotify.com/auth/token',
           'data':{'username':'test','password':'dummy','time':datetime(2023,1,2), 'options':[1,True,None]}}
    assert diagnostic_json(value)==jsonable_encoder(value)


def test_unsupported_arguments_do_not_change_live_requests(tmp_path):
    class BadRepr:
        def __repr__(self):
            raise RuntimeError('logging must not invoke this')
    cycle=[];cycle.append(cycle)
    args={'placeholder':Ellipsis,'bad_bytes':b'\xff','complex':1+2j,'nan':float('nan'),
          'surrogate':'\ud800','cycle':cycle,'opaque':BadRepr(),'not_implemented':NotImplemented}
    tracker=SimpleNamespace(requests=[{'method':'post','url':'fake','data':args}])
    path=tmp_path/'api_calls.jsonl';save_request_log(tracker,path)
    record=json.loads(path.read_text())
    assert record['data']['placeholder']=={'__python_type__':'ellipsis','value':'...'}
    assert record['data']['surrogate']=='\ud800'
    assert tracker.requests[0]['data'] is args and args['placeholder'] is Ellipsis
    assert args['cycle'][0] is cycle and isinstance(args['opaque'],BadRepr)
    json.dumps(record,allow_nan=False)
