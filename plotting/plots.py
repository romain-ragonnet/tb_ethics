from matplotlib import pyplot as plt


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

    labels = [intervention_names[dec_var.split("decision_var_")[1]] for dec_var in  opti_mle_params.keys()]
    values = opti_mle_params.values()

    ax.pie(values, labels=labels, explode=[0.05]*3)
    ax.set_title("Optimal intervention plan")
    
    return ax


def plot_future_trajectories(opti_bcm, opti_mle_params, output="incidence_per100k", xmin=2020, ax=None):
    baseline_decision_vars = {
        "decision_var_trans": 0.,
        "decision_var_cdr": 0.,
        "decision_var_pt": 0.
    }

    if not ax:
        fig, ax = plt.subplots(1, 1)
    
    ymax = 0.
    for sc_name, dec_vars in zip(['baseline', 'optimised'], [baseline_decision_vars, opti_mle_params]):
        res = opti_bcm.run(dec_vars)
        derived_df = res.derived_outputs
        derived_df[output].loc[xmin:].plot(label=sc_name)
        ymax = max(ymax, derived_df[output].loc[xmin:].max())

    ax.set_ylabel(output_names[output])
    ax.set_ylim((0, 1.2 * ymax))
    ax.legend()

    return ax
    