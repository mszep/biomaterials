import itertools
import os
import numpy as np
from pymatgen.matproj.rest import MPRester
import sklearn as skl
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import LeaveOneOut

materials = {"a-C:H": {"formula": ["C", "H"], "score": .45-.52},
             "Si:a-C:H": {"formula": ["Si", "C", "H"], "score": .90-.86},
             "Ti:a-C:H": {"formula": ["Ti", "C"], "score": .63-.44},
             "TiN": {"formula": ["Ti", "N"], "score": .74-.53},
             "TiOx": {"formula": ["Ti", "O"], "score": .42-.76},
             "Ti": {"formula": ["Ti"], "score": .45-.70}}

prediction_elements = ["H", "C", "N", "O", "Si", "P", "S",
                       "Sc", "Ti", "V"]

properties = ["band_gap", "e_above_hull",
              "energy_per_atom", "formation_energy_per_atom"]

def get_element_list(materials):
    """Given a dict of materials such as the one defined above, return
    the set of elements that they contain. E.g. a dict containing only
    TiN and TiOx would return ["Ti", "O", "N"]."""

    # extract the list of constituent elements from each dict entry
    element_lists = [materials[key]["formula"] for key in materials]
    # concatenate the lists and use set() to remove duplicates
    return list(set(itertools.chain(*element_lists)))

def get_data(elements):
    m = MPRester(os.environ['MPROJ_API_KEY'])
    entries = m.get_entries_in_chemsys(elements)
    return entries

def filter_entries(entries, elements):
    for e in entries:
        formula = e.data["unit_cell_formula"]
        if ( len(formula) == len(elements) and
           all(el in formula for el in elements)):
            yield e

def get_average_property(entries, prop):
    return np.mean([e.data[prop] for e in entries])

def get_training_data():
    elements = get_element_list(materials)
    entries = get_data(elements)
    Xraw = np.zeros((len(materials), len(properties)))
    Y = np.zeros(len(materials))

    for i, c in enumerate(materials):
        Y[i] = materials[c]["score"]
        filtered_data = list(filter_entries(entries, materials[c]["formula"]))
        for j, p in enumerate(properties):
            Xraw[i,j] = get_average_property(filtered_data, p)

    scaler = skl.preprocessing.StandardScaler().fit(Xraw)
    return scaler.transform(Xraw), Y, scaler

def get_model(X, Y):
    return linear_model.LinearRegression().fit(X, Y)

def train_model(materials):
    X, Y, scaler = get_training_data(training_entries, materials)
    model = get_model(X, Y)
    return X, Y, model, scaler

def score_material(entry, model, scaler):
    Xraw = np.array([entry.data[prop] for prop in properties])
    return model.predict(scaler.transform(Xraw))

def find_best_materials(entries, model, scaler):
    entries.sort(key=lambda e: score_material(e, model, scaler),
                 reverse=True)
        
def main(intup=None):
    if intup == None:
        X, Y, scaler = get_training_data()
    else:
        X, Y, scaler = intup

    mod = linear_model.LinearRegression()
    # First, we cross-validate using leave-one-out
    cv = LeaveOneOut(X.shape[0])
    scores = cross_validation.cross_val_score(mod, X, Y, cv=cv, scoring='mean_absolute_error')
    print "Cross-validation scores: ", scores
    model = mod.fit(X, Y)

    prediction_entries = get_data(prediction_elements)
    find_best_materials(prediction_entries, model, scaler)
    print "30 best materials according to a linear model:"
    for i in range(30):
        print prediction_entries[i].data['pretty_formula'],\
              score_material(prediction_entries[i], model, scaler)

    return X, Y, scaler
