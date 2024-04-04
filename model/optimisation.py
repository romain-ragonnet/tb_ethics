import pandas as pd

from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est

from model.model import build_model


def get_optimisation_bcm(fixed_params, decision_var_sum_threshold= 1.5) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model object for the purpose of intervention optimisation.
    """
    model = build_model(fixed_params)

    priors =  [esp.UniformPrior(f"dec_var_{intervention}", (0, 1.)) for intervention in ['trans', 'cdr', 'pt']]

    targets = [
        est.NormalTarget("incidence_per100k", pd.Series({2040: 0.}), stdev=10.) # used to minimise indicence
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

    return BayesianCompartmentalModel(model, fixed_params, priors, targets)
