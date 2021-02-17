# This file corrects a bug in the Keras Surgeon repo where they do not
# handle list outputs or dict outputs from a model correctly

class TensorKeys(list):
    def __init__(self, refs):
        super().__init__(refs)

    def __contains__(self, item):
        if isinstance(item, dict):
            for v in item.values():
                if super(TensorKeys, self).__contains__(v.ref()):
                    return True
            return False
        elif isinstance(item, list):
            for v in item:
                if super(TensorKeys, self).__contains__(v.ref()):
                    return True
            return False

        try:
            return super().__contains__(item.ref())
        except AttributeError:
            return super().__contains__(item.experimental_ref())


class TensorDict(dict):
    def __init__(self):
        super().__init__()

    def __setitem__(self, key, value):
        if isinstance(key, dict):
            for v in key.values():
                super(TensorDict, self).__setitem__(v.ref(), value)
            return
        elif isinstance(key, list):
            for v in key:
                super(TensorDict, self).__setitem__(v.ref(), value)
            return

        try:
            super().__setitem__(key.ref(), value)
        except AttributeError:
            super().__setitem__(key.experimental_ref(), value)

    def __getitem__(self, item):
        if isinstance(item, dict):
            try:
                return {k: super(TensorDict, self).__getitem__(v.ref()) for k, v in item.items()}
            except:
                return {}
        elif isinstance(item, list):
            try:
                return [super(TensorDict, self).__getitem__(v.ref()) for v in item]
            except:
                return []

        try:
            return super().__getitem__(item.ref())
        except AttributeError:
            return super().__getitem__(item.experimental_ref())

    def keys(self):
        return TensorKeys(super().keys())

import kerassurgeon.surgeon
kerassurgeon.surgeon.TensorDict = TensorDict
