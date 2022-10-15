import numpy as np
from itertools import combinations, chain


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def get_trans_name(row):
    if (row['TransformationType'] == "log(adstock/c+1)") | (row['TransformationType'] == "log(adstock/c)"):
        row['Scalar'] = "K_" + row['BaseVarName']
        val = str(int(10 * row['Decay'])) + str(int(row['Peak'])) + str(int(row['Length']))
        row['adNames'] = val
        row['TransformedVarName'] = row['BaseVarName'] + "_" + val
    elif (row['TransformationType'] == "log(adstock+1)") | (row['TransformationType'] == "log(adstock)") | (
            row['TransformationType'] == "adstock"):
        val = str(int(10 * row['Decay'])) + str(int(row['Peak'])) + str(int(row['Length']))
        row['adNames'] = val
        row['TransformedVarName'] = row['BaseVarName'] + "_" + val
    elif (row['TransformationType'] == "log(x/c+1)") | (row['TransformationType'] == "log(x/c)"):
        row['Scalar'] = "K_" + row['BaseVarName']
        row['TransformedVarName'] = row['BaseVarName'] + "_LOGC"
    elif (row['TransformationType'] == "log(x+1)") | (row['TransformationType'] == "log(x)"):
        row['TransformedVarName'] = row['BaseVarName'] + "_LOGX"
    else:
        row['TransformedVarName'] = row['BaseVarName']
    return row


def compute_adstock_vector(dim, params):
    wk = 0
    l, p, r = params[0], params[1], params[2]
    y = []
    if p > 0:
        c = 1.0 / (1.0 + p)
    else:
        c = 0.0
    for i in range(dim):
        if wk <= l:
            if (i < p) & (p > 0):
                y.append(i / (p + 1) + c)
            else:
                y.append(np.power(r, i - p))
        else:
            y.append(0.0)
        wk += 1

    y = np.array(y) / np.sum(y)
    return y


def get_adstock_matrix(dim, asvals_53wk):
    adstock_matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(len(asvals_53wk)):
            if (i + j) < dim:
                adstock_matrix[i + j, i] = asvals_53wk[j]
            else:
                break
    return adstock_matrix


def gen_adstock_operator(var, adstock_params, model_duration):
    S = get_adstock_matrix(model_duration, compute_adstock_vector(model_duration, adstock_params[var]))
    return S


def transform(pdf, transpec, Ses):
    pdf = pdf.sort_values(by=['RowIndex'])
    c_len = len(pdf)
    for i, row in transpec.iterrows():
        tType = row['TransformationType']
        bName = row['BaseVarName']
        tName = row['TransformedVarName']
        adName = row['adNames']
        scName = row['Scalar']
        v = pdf[bName]
        if tType == 'log(adstock/c+1)':
            pdf[tName] = np.log(np.divide(np.matmul(Ses.get(adName)[0:c_len, 0:c_len], v), pdf[scName]) + 1.0)
        elif tType == 'log(adstock/c)':
            pdf[tName] = np.log(np.divide(np.matmul(Ses.get(adName)[0:c_len, 0:c_len], v), pdf[scName]))
        elif tType == 'log(adstock+1)':
            pdf[tName] = np.log(np.matmul(Ses.get(adName)[0:c_len, 0:c_len], v) + 1.0)
        elif tType == 'log(adstock)':
            pdf[tName] = np.log(np.matmul(Ses.get(adName)[0:c_len, 0:c_len], v))
        elif tType == 'adstock':
            pdf[tName] = np.matmul(Ses.get(adName)[0:c_len, 0:c_len], v)
        elif tType == 'log(x/c+1)':
            pdf[tName] = np.log(np.divide(v, pdf[scName]) + 1.0)
        elif tType == 'log(x/c)':
            pdf[tName] = np.log(np.divide(v, pdf[scName]))
        elif tType == 'log(x+1)':
            pdf[tName] = np.log(v + 1.0)
        elif tType == 'log(x)':
            pdf[tName] = np.log(v)
        l = int(row['Lag'])
        if (l > 0) & (l < len(pdf)):
            pdf[tName] = pdf[tName].shift(l).fillna(0.0)
    return pdf


def is_nan(s):
    return s != s


# to decode the number generated by nevergrad
def get_combo_dict(n=8, mvar=3):
    dd = dict()
    gvars = [x + 1 for x in range(n)]
    for i in gvars:
        cl = list()
        for j in range(mvar + 1):
            if j <= i:
                cl.append(list(combinations(list(range(i)), j)))
                if len(cl) > 0:
                    ccl = list(chain(*cl))
                    cd = dict(zip(list(range(len(ccl))), ccl))
                    dd.update({str(i) + "_" + str(j): cd})
    return dd
