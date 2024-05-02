import yaml
from matplotlib import pyplot as plt
from model.calibration import FIXED_PARAMS_PATH
from model.optimisation import get_optimisation_bcm

output_names = {
    "incidence_per100k": "TB incidence (/100,000/y)",
    "cumulative_incidence": "Cumulative TB episodes",
    "ltbi_prevalence_perc": "LTBI prevalence (%)",
    "cumulative_future_deaths": "Cumulative TB deaths",
    "cumulative_future_paed_deaths": "Cumulative paediatric TB deaths",
    "infection_per100k": "infection rate (/100k/y)",
    "tb_deaths": "TB mortality",
    "prop_incidence_early": "Incidence prop. from early activation",
    "ltbi_prevalence_early_perc": "Early LTBI prevalence (%)",
    "ltbi_prevalence_late_perc": "Late LTBI prevalence (%)",
}

intervention_names = {
    "cdr": "Improved detection",
    "pt": "Preventive treatment",
    "trans": "Reduced transmission"
}

def make_intervention_piechart(opti_mle_params, ax=None):
    if not ax:
        fig, ax = plt.subplots()

    labels = [intervention_names[dec_var.split("decision_var_")[1]] for dec_var in list(opti_mle_params.keys())] 
    values = list(opti_mle_params.values())

    ax.pie(values, labels=labels, explode=[0.05]*3)
    ax.set_title("Optimal intervention plan")
    
    return ax


def plot_future_trajectories(opti_bcm, opti_mle_params, output="incidence_per100k", xmin=2020, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)    

    baseline_decision_vars = {f"decision_var_{intervention}": 0. for intervention in ['trans', 'cdr', 'pt']}

    ymax = 0.
    for sc_name, dec_vars in zip(['baseline', 'optimised'], [baseline_decision_vars, opti_mle_params]):
        res = opti_bcm.run(dec_vars)
        derived_df = res.derived_outputs
        derived_df[output].loc[xmin:].plot(label=sc_name, ax=ax)
        ymax = max(ymax, derived_df[output].loc[xmin:].max())

    ax.set_ylabel(output_names[output])
    ax.set_ylim((0, 1.2 * ymax))
    ax.legend()

    return ax
    