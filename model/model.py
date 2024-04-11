from jax import numpy as jnp

from summer2 import CompartmentalModel, AgeStratification, Overwrite, Multiply
from summer2.parameters import Parameter, DerivedOutput, Function
from summer2.functions import time as stf


AGEGROUPS = ["0", "5", "15", "40", "60", "75"]


def build_model(fixed_params: dict):

    compartments = ["S", "E1", "E2", "I"]
    model = CompartmentalModel(
        times=(fixed_params['time_start'], fixed_params['time_end']),
        compartments=compartments,
        infectious_compartments=["I"],
        timestep=fixed_params['time_step'],
    )

    model.set_initial_population({"S": fixed_params['start_population'] - 1., "I": 1.})

    # Birth and background mortality
    model.add_crude_birth_flow("birth", fixed_params["crude_birth_rate"], "S")
    model.add_universal_death_flows("universal_death", 1.0) # later adjusted by age

    # Transmission flows
    if fixed_params["use_interventions"]:
        future_transmission_multiplier = 1. - Parameter("decision_var_trans") * fixed_params["max_transmission_reduction"] 
        tv_transmission_adj = stf.get_linear_interpolation_function(
            x_pts = [fixed_params["intervention_time"], fixed_params["intervention_time"] + 1.], 
            y_pts = [1., future_transmission_multiplier]
        )    
        model.add_infection_frequency_flow(name="infection", contact_rate=fixed_params["fitted_effective_contact_rate"] * tv_transmission_adj, source="S", dest="E1")
        model.add_infection_frequency_flow(name="reinfection", contact_rate=fixed_params["fitted_effective_contact_rate"] * tv_transmission_adj * fixed_params['rr_reinfection'], source="E2", dest="E1")
    else:
        model.add_infection_frequency_flow(name="infection", contact_rate=Parameter("effective_contact_rate"), source="S", dest="E1")    
        model.add_infection_frequency_flow(name="reinfection", contact_rate=Parameter("effective_contact_rate") * fixed_params['rr_reinfection'], source="E2", dest="E1")

    # Latency progression flows (all progression rates set to 1, but later adjusted by age)
    model.add_transition_flow(name="stabilisation", fractional_rate=1., source="E1", dest="E2")
    model.add_transition_flow(name="early_activation", fractional_rate=1., source="E1", dest="I")
    model.add_transition_flow(name="late_activation", fractional_rate=1., source="E2", dest="I")

    # Recovery flows
    model.add_transition_flow(name="self_recovery", fractional_rate=fixed_params["self_recovery_rate"], source="I", dest="S")    
    model.add_transition_flow(name="tx_recovery", fractional_rate= 1., source="I", dest="S")  # later adjusted by age

    # TB death
    model.add_death_flow("tb_death", fixed_params["tb_mortality_rate"], "I")

    # Preventive treatment
    if fixed_params["use_interventions"]:
        future_pt_rate = Parameter("decision_var_pt") * fixed_params["max_pt_rate"]
        for comp in ["E1", "E2"]:
            pt_rate = stf.get_linear_interpolation_function(
                x_pts = [fixed_params["intervention_time"], fixed_params["intervention_time"] + 1.], 
                y_pts = [0., future_pt_rate]
            )
            model.add_transition_flow(name=f"pt_{comp}", fractional_rate=pt_rate, source=comp, dest="S")    

    # Stratification by age
    stratify_model_by_age(model, fixed_params, compartments)

    # Outputs
    request_model_outputs(model, compartments, fixed_params['intervention_time'], fixed_params['life_expectancy'])

    return model


def stratify_model_by_age(model, fixed_params, compartments):
    strat = AgeStratification("age", AGEGROUPS, compartments)

    # Adjust progression flows
    for flow_name in ["early_activation", "late_activation", "stabilisation"]:
        # add missing parameter values
        for age in AGEGROUPS[3:]:
            fixed_params[f"{flow_name}_rate"][age] = fixed_params[f"{flow_name}_rate"]["15"]

        adjs = {
            age: Overwrite(fixed_params[f"{flow_name}_rate"][age])  for age in AGEGROUPS
        }

        strat.set_flow_adjustments(flow_name, adjs)

    # Adjust background mortality rates
    mort_adjs = {age: Overwrite(fixed_params["background_mortality_rate"][age]) for age in AGEGROUPS}
    strat.set_flow_adjustments("universal_death", mort_adjs)

    # Adjust detection/treatment rates
    if fixed_params["use_interventions"]:
        future_cdr = fixed_params["CDR_2000"] + Parameter("decision_var_cdr") * (fixed_params["max_intervention_cdr"] - fixed_params["CDR_2000"])
    else:
        future_cdr = fixed_params["CDR_2000"]
    cdr = stf.get_linear_interpolation_function(
        x_pts = [1950., 2000., fixed_params["intervention_time"], fixed_params["intervention_time"] + 1.], 
        y_pts = [0., fixed_params["CDR_2000"], fixed_params["CDR_2000"], future_cdr]
    )
    detect_adjs = {age: Overwrite(fixed_params["TSR"] * cdr * (fixed_params["background_mortality_rate"][age] + fixed_params["tb_mortality_rate"] + fixed_params["self_recovery_rate"]) / (1 - cdr)) for age in AGEGROUPS}
    strat.set_flow_adjustments("tx_recovery", detect_adjs)

    # Make kids non-infectious
    inf_adjs = {age: Multiply(0.) if age in ["0", "5"] else Multiply(1.0) for age in AGEGROUPS}
    strat.add_infectiousness_adjustments("I", inf_adjs)

    model.stratify_with(strat)


def request_model_outputs(model, compartments, intervention_time, life_expectancy):

    model.request_output_for_compartments("total_population", compartments)

    # incidence outputs
    model.request_output_for_flow("incidence_early_raw", "early_activation", save_results=False)
    model.request_output_for_flow("incidence_late_raw", "late_activation", save_results=False)
    model.request_function_output("incidence_per100k", 1.e5 * (DerivedOutput("incidence_early_raw") + DerivedOutput("incidence_late_raw")) / DerivedOutput("total_population"), save_results=True)

    # prevalence of latent and active TB
    model.request_output_for_compartments("ltbi_prevalence", ["E1", "E2"], save_results=False)
    model.request_function_output("ltbi_prevalence_perc", 100 * DerivedOutput("ltbi_prevalence") / DerivedOutput("total_population"), save_results=True)
    model.request_output_for_compartments("tb_prevalence", ["I"], save_results=False)
    model.request_function_output("tb_prevalence_per100k", 1.e5 * DerivedOutput("tb_prevalence") / DerivedOutput("total_population"), save_results=True)

    # death outputs
    model.request_output_for_flow(f"tb_deaths", "tb_death")
    for age in AGEGROUPS:
        model.request_output_for_flow(f"tb_deathsXage_{age}", "tb_death", source_strata={"age": age})

    # cumulative death outputs (overall and paediatric)
    model.request_cumulative_output(name=f"cumulative_future_deaths", source=f"tb_deaths", start_time=intervention_time, save_results=True)

    paed_agegroups = ["0", "5"]
    for age in AGEGROUPS:
        model.request_cumulative_output(name=f"cumulative_future_deathsXage_{age}", source=f"tb_deathsXage_{age}", start_time=intervention_time, save_results=False)
    model.request_aggregate_output(name="cumulative_future_paed_deaths", sources=[f"cumulative_future_deathsXage_{age}" for age in paed_agegroups], save_results=True)

    # years of life lost
    for age in AGEGROUPS:
        model.request_function_output(name=f"years_of_life_lostXage_{age}", func=DerivedOutput(f"cumulative_future_deathsXage_{age}") * life_expectancy[age], save_results=False)
    model.request_aggregate_output(name="years_of_life_lost", sources = [f"years_of_life_lostXage_{age}" for age in AGEGROUPS], save_results=True)

    # track sum of decision variables
    def repeat_val(example_output, value):
        return jnp.repeat(value, jnp.size(example_output))

    decision_var_sum_func = Function(repeat_val, [DerivedOutput("total_population"), Parameter("decision_var_trans") + Parameter("decision_var_cdr") + Parameter("decision_var_pt")])
    model.request_function_output("decision_var_sum", decision_var_sum_func, save_results=True)
