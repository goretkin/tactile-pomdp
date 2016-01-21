import numpy as np
def quiet_log(x):
    # we want log(0) = -inf without any warning or error.
    a = np.seterr(divide="ignore")
    r = np.log(x)
    np.seterr(divide=a["divide"])
    return r

class silent_division_by_zero_class:
    def __enter__(self):
        self.a = np.seterr(divide="ignore")

    def __exit__(self, exc_type, exc_value, traceback):
        np.seterr(divide=self.a["divide"])

silent_division_by_zero = silent_division_by_zero_class()
