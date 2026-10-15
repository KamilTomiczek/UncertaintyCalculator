from scipy import stats
import numpy as np
import math

def getStudentTFactor(N, confidence_level):
    df = N - 1
    q = (1 + confidence_level) / 2

    return round(stats.t.ppf(q, df), 3)

def uncertaintyTypeA(measurement, confidence_level=0.6826):

    n = len(measurement)
    if n < 2:
        raise ValueError("At least two measurements are required to calculate type A uncertainty.")
    
    mean_value = sum(measurement) / n

    type_a_uncertainty = ((sum((x - mean_value) ** 2 for x in measurement) / (n * (n - 1))) ** 0.5) * getStudentTFactor(n, confidence_level)
    
    return mean_value, type_a_uncertainty

def uncertaintyTypeB(delta_max):
    return delta_max / math.sqrt(3)

def deltaMaxAnalog(instrument_class, instrument_range):
    delta_max = (instrument_class * instrument_range) / 100.0
    return delta_max

def deltaMaxDigital(wynik, instrument_class, b_dgt, resolution):
    delta_max = (instrument_class / 100.0) * abs(wynik) + (b_dgt * resolution)
    return delta_max

def uncertaintyTypeBCounting(N):
    return math.sqrt(N)

def combinedUncertainty(type_a, type_b):
    return math.hypot(type_a, type_b)

def uncertaintyPropagatedDifference(u1, u2):
    return math.hypot(u1, u2)

def linearRegression(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    N = len(x)
    
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x**2)
    Sxy = np.sum(x * y)
    
    denominator = N * Sxx - Sx**2
    if denominator == 0 or N < 3:
        return np.nan, np.nan, np.nan, np.nan
        
    a = (N * Sxy - Sx * Sy) / denominator
    b = (Sxx * Sy - Sx * Sxy) / denominator
    
    epsilon = y - (a * x + b)
    S_eps = np.sum(epsilon**2)
    
    u_a = np.sqrt((N / (N - 2)) * S_eps / denominator)
    u_b = np.sqrt((1 / (N - 2)) * Sxx * S_eps / denominator)
    
    return a, b, u_a, u_b

def round_to_sig_figs(val, sig_figs=2):
    if val == 0 or math.isnan(val) or math.isinf(val):
        return 0.0, 0
    digits = -int(math.floor(math.log10(abs(val)))) + (sig_figs - 1)
    rounded_val = round(val, digits)
    return rounded_val, max(0, digits)

def format_result(val, unc, sig_figs=2):

    if math.isnan(val) or math.isnan(unc):
        return "Brak danych"
    unc_rounded, decimals = round_to_sig_figs(unc, sig_figs)
    val_rounded = round(val, decimals)
    if decimals == 0:
        return f"{int(val_rounded)} ± {int(unc_rounded)}"
    return f"{val_rounded:.{decimals}f} ± {unc_rounded:.{decimals}f}"