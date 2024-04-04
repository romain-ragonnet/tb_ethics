import pandas as pd
import yaml
from pathlib import Path
from matplotlib import pyplot as plt

from estival.wrappers.nevergrad import optimize_model
from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est

from model.model import build_model

FIXED_PARAMS_PATH = Path(__file__).parent.resolve() / "fixed_params.yml"


def get_calibration_bcm(fixed_params, target_incidence=100) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model object for the purpose of model calibration.
    """
    model = build_model(fixed_params)

    priors =  [
        esp.UniformPrior("effective_contact_rate", (2., 100.)), 
    ]

    targets = [
        est.NormalTarget("incidence_per100k", pd.Series({2024: target_incidence}), stdev=10.)
    ]

    return BayesianCompartmentalModel(model, fixed_params, priors, targets)


def fit_model(target_incidence=200): 
    with open(FIXED_PARAMS_PATH, "r") as f:
        fixed_params = yaml.safe_load(f)
    bcm = get_calibration_bcm(fixed_params, target_incidence=200)    
    orunner = optimize_model(bcm, num_workers=8)
    rec = orunner.minimize(1000)
    mle_params = rec.value[1]

    return bcm, mle_params


def check_fit(bcm, mle_params):
    baseline_decision_vars = {
        "decision_var_trans": 0.,
        "decision_var_cdr": 0.,
        "decision_var_pt": 0.
    }
    res = bcm.run(mle_params | baseline_decision_vars)
    derived_df = res.derived_outputs
    derived_df['incidence_per100k'].plot()

    target_data = bcm.targets['incidence_per100k'].data
    plt.scatter(target_data.index, target_data, color='red')