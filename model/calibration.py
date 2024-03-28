import pandas as pd

from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est

from model.model import build_model


def get_bcm(fixed_params, target_incidence=100) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model object.
    """
    model = build_model(fixed_params)

    priors =  [
        esp.UniformPrior("effective_contact_rate", (2., 100.)), 
    ]

    targets = [
        est.NormalTarget("incidence_per100k", pd.Series({2024: target_incidence}), stdev=10.)
    ]

    return BayesianCompartmentalModel(model, fixed_params, priors, targets)
