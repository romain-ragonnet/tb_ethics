import pandas as pd
from jax import numpy as jnp
import yaml

from estival.wrappers.nevergrad import optimize_model
from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est

from model.model import build_model
from model.calibration import FIXED_PARAMS_PATH


def get_optimisation_bcm(interv_params, decision_var_sum_threshold= 1.5, minimised_indicator="incidence_per100k") -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model object for the purpose of intervention optimisation.
    """
    model = build_model(interv_params)

    priors =  [esp.UniformPrior(f"decision_var_{intervention}", (0, 1.)) for intervention in ['trans', 'cdr', 'pt']]

    targets = [
        est.NormalTarget(minimised_indicator, pd.Series({2040: 0.}), stdev=10.) # used to minimise indicence
    ]

    # define another target to censor decision variables above a given threshold

    def censored_func(modelled, data, parameters, time_weights):     
        # Returns a very large negative number if modelled value is greater than threshold. Returns 0 otherwise.
        return jnp.where(modelled > decision_var_sum_threshold, -1.e11, 0.)[0]

    targets.append(
        est.CustomTarget(
            "decision_var_sum", 
            pd.Series([0.], index=[2040.]), # could be any value, only the time index matters
            censored_func
        )
    )

    return BayesianCompartmentalModel(model, interv_params, priors, targets)


def optimise_interventions(mle_params, decision_var_sum_threshold=1.5, minimised_indicator="incidence_per100k"):
    with open(FIXED_PARAMS_PATH, "r") as f:
        fixed_params = yaml.safe_load(f)

    interv_params = fixed_params | {"use_interventions":True, "fitted_effective_contact_rate": mle_params["effective_contact_rate"]}
    opti_bcm = get_optimisation_bcm(interv_params, decision_var_sum_threshold, minimised_indicator)

    opti_orunner = optimize_model(opti_bcm, num_workers=8)
    opti_rec = opti_orunner.minimize(1000)
    opti_mle_params = opti_rec.value[1]

    return opti_bcm, opti_mle_params