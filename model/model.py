from summer2 import CompartmentalModel, AgeStratification, Overwrite, Multiply
from summer2.parameters import Parameter, DerivedOutput
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

    # Stratification by age
    stratify_model_by_age(model, fixed_params, compartments)

    # Outputs
    request_model_outputs(model, compartments)

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
    future_cdr = .80  #FIXME: placeholder only for now
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


def request_model_outputs(model, compartments):

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
