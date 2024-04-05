import yaml
from matplotlib import pyplot as plt
from model.calibration import FIXED_PARAMS_PATH
from model.optimisation import get_optimisation_bcm

output_names = {
    "incidence_per100k": "TB incidence (/100k/y)"
}

intervention_names = {
    "cdr": "Improved detection",
    "pt": "Preventive treatment",
    "trans": "Reduced transmission"
}

def make_intervention_piechart(opti_mle_params, ax=None):
    if not ax:
        fig, ax = plt.subplots()

    # retrieve maximum intervention sum from fixed parameters
    with open(FIXED_PARAMS_PATH, "r") as f:
        fixed_params = yaml.safe_load(f)
    decision_var_sum_threshold = fixed_params["max_intervention_sum"]

    labels = [intervention_names[dec_var.split("decision_var_")[1]] for dec_var in list(opti_mle_params.keys()) + ["decision_var_pt"]] 
    values = list(opti_mle_params.values()) + [decision_var_sum_threshold - sum(list(opti_mle_params.values()))]

    ax.pie(values, labels=labels, explode=[0.05]*3)
    ax.set_title("Optimal intervention plan")
    
    return ax


def plot_future_trajectories(opti_bcm, opti_mle_params, fitted_effective_contact_rate, output="incidence_per100k", xmin=2020, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    
    with open(FIXED_PARAMS_PATH, "r") as f:
        fixed_params = yaml.safe_load(f)
    baseline_params = fixed_params | {
        "use_interventions":True, "fitted_effective_contact_rate": fitted_effective_contact_rate, "max_intervention_sum": 0.}
    baseline_bcm = get_optimisation_bcm(baseline_params, 0.)
    baseline_decision_vars = {
        'decision_var_trans': 0.,
        'decision_var_cdr': 0.
    }

    ymax = 0.
    for sc_name, dec_vars in zip(['baseline', 'optimised'], [baseline_decision_vars, opti_mle_params]):
        bcm = baseline_bcm if sc_name == 'baseline' else opti_bcm        
        res = bcm.run(dec_vars)
        derived_df = res.derived_outputs
        derived_df[output].loc[xmin:].plot(label=sc_name)
        ymax = max(ymax, derived_df[output].loc[xmin:].max())

    ax.set_ylabel(output_names[output])
    ax.set_ylim((0, 1.2 * ymax))
    ax.legend()

    return ax
    