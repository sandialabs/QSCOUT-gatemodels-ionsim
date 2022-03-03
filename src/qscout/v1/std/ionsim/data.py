import itertools

GATE_NAMES = ['R', 'MS']
_gate_data = {}

class GateData(dict):
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params

    @classmethod
    def params_str(klass, params):
        return '_'.join([str(p) for p in params])

    def describe(self):
        descr = {}
        for name, gate in self.items():
            descr[name] = dict(zip(
                ['phi', 'theta', *self.params],
                gate.opfactory.base_interpolator.parameter_ranges)
            )

        return descr

def get_gate_data(model_name, params):
    parent_path = __name__[:-5]
    params_str = GateData.params_str(params)

    try:
        return _gate_data[(model_name, params_str)]
    except KeyError:
        pass

    data = _gate_data[(model_name, params_str)] = GateData(model_name, params)

    import pkgutil, dill
    for name in GATE_NAMES:
        raw = pkgutil.get_data(parent_path, f"{model_name}/{name}_phi_theta_{params_str}.pyg")
        data[name] = dill.loads(raw)

    return data
