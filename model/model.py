from summer2 import CompartmentalModel, AgeStratification, Overwrite
from summer2.parameters import Parameter, DerivedOutput


def build_model(fixed_params: dict):

    compartments = ["S", "E1", "E2", "I"]
    model = CompartmentalModel(
        times=(fixed_params['time_start'], fixed_params['time_end']),
        compartments=compartments,
        infectious_compartments=["I"],
        timestep=fixed_params['time_step'],
    )

    model.set_initial_population({"S": fixed_params['start_population'] - 1., "I": 1.})

    # Latency progression flows (all progression rates set to 1, but later adjusted by age)
    model.add_transition_flow(name="stabilisation", fractional_rate=1., source="E1", dest="E2")
    model.add_transition_flow(name="early_activation", fractional_rate=1., source="E1", dest="I")
    model.add_transition_flow(name="late_activation", fractional_rate=1., source="E2", dest="I")

    # Transmission flows
    model.add_infection_frequency_flow(name="infection", contact_rate=Parameter("effective_contact_rate"), source="S", dest="E1")
    model.add_infection_frequency_flow(name="reinfection", contact_rate=Parameter("effective_contact_rate") * fixed_params['rr_reinfection'], source="E2", dest="E1")

    # Recovery flows
    model.add_transition_flow(name="self_recovery", fractional_rate=fixed_params["self_recovery_rate"], source="I", dest="S")    
    model.add_transition_flow(name="tx_recovery", fractional_rate= 1., source="I", dest="S")  # later adjusted by age

    # Death
    model.add_universal_death_flows("universal_death", 1.0) # later adjusted by age
    model.add_death_flow("tb_death", fixed_params["tb_mortality_rate"], "I")

    # Birth
    model.add_crude_birth_flow("birth", fixed_params["crude_birth_rate"], "S")

    # Stratification by age
    stratify_model_by_age(model, fixed_params, compartments)

    # Outputs
    request_model_outputs(model, compartments)

    return model


def stratify_model_by_age(model, fixed_params, compartments):
    agegroups = ["0", "5", "15", "40", "60", "75"]
    strat = AgeStratification("age", agegroups, compartments)

    # Adjust progression flows
    for flow_name in ["early_activation", "late_activation", "stabilisation"]:
        # add missing parameter values
        for age in agegroups[3:]:
            fixed_params[f"{flow_name}_rate"][age] = fixed_params[f"{flow_name}_rate"]["15"]

        adjs = {
            age: Overwrite(fixed_params[f"{flow_name}_rate"][age])  for age in agegroups
        }

        strat.set_flow_adjustments(flow_name, adjs)

    # Adjust background mortality rates
    mort_adjs = {age: Overwrite(fixed_params["background_mortality_rate"][age]) for age in agegroups}
    strat.set_flow_adjustments("universal_death", mort_adjs)


    # Adjust detection/treatment rates
    TSR = .85
    cdr = fixed_params["CDR"] 
    detect_adjs = {age: Overwrite(TSR * cdr * (fixed_params["background_mortality_rate"][age] + fixed_params["tb_mortality_rate"] + fixed_params["self_recovery_rate"]) / (1 - cdr)) for age in agegroups}
    strat.set_flow_adjustments("tx_recovery", detect_adjs)

    model.stratify_with(strat)


def request_model_outputs(model, compartments):

    model.request_output_for_compartments("total_population", compartments)

    model.request_output_for_flow("incidence_early_raw", "early_activation", save_results=False)
    model.request_output_for_flow("incidence_late_raw", "late_activation", save_results=False)
    model.request_function_output("incidence_per100k", 1.e5 * (DerivedOutput("incidence_early_raw") + DerivedOutput("incidence_late_raw")) / DerivedOutput("total_population"), save_results=True)
