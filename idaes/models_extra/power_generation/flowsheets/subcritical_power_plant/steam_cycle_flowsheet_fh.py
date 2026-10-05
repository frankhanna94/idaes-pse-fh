#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Steady state sub-flowsheet for a subcritical steam cycle system.

This flowsheet is a scale up of a former one. The original flowsheet 
refers to a 300MWe steam cycle system. This flowsheet models a 660 MWe
steam cycle system.

Source flowsheet: 
    GitHub Repository:  https://github.com/IDAES/idaes-pse.git
    Path:               idaes/models_extra/power_generation/flowsheets/subcritical_power_plant
                        /steam_cycle_flowsheet.py

Approach, Assumptions, and 
* High level flow scaling factor estimated at 2.2 (target capacity/original capacity)
* Unit models scaling factors maintained as is from the original model. 
    Rationale: none of the results/outcomes of the analysis varies by an order of
                magnitude.
* Started the modeling from the feedwater heaters
    ** Each fwh was modeled separately before merging into a single flowsheet
    ** The overall heat transfer coefficient was maintained the same while solving 
        for the area
    ** In the full flowsheet, A and U are fixed; the model solves for the extraction
        steam requirement, ensuring the following:
            *** the inlet flow is fixed (2.2 x original flow)
            *** the cold side inlet temperature (at fwh1) and outlet temperature 
                (at fwh6) match (approximately) the original model

Status
* 06/25/2026 - This flowsheet is limited to the feedwater heaters FWH1--> FWH6

"""

# Import dependencies
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

from idaes.models_extra.power_generation.unit_models.helm import (
    HelmMixer as Mixer,
    MomentumMixingType,
    HelmValve as Valve,
    HelmIsentropicCompressor as WaterPump,
    HelmSplitter as Separator
)

from idaes.models_extra.power_generation.unit_models import ( 
    WaterTank,
    FWH0DDynamic as FWH0D,
)

from idaes.models.properties import iapws95

from idaes.core import FlowsheetBlock
from idaes.core.util.initialization import propagate_state as _set_port
from idaes.core.util.initialization import fix_state_vars
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale
from idaes.core.solvers import get_solver

import idaes.logger as idaeslog

def main_steady_state():
    '''
    Main function to run the steady state model
    '''
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.prop_water = iapws95.Iapws95ParameterBlock()
    m.fs.stc = FlowsheetBlock(time_units=pyo.units.s)
    
    m = add_unit_models(m)
    m = set_arcs_and_constraints(m)
    m = set_inputs(m)
    set_scaling_factors(m)
    
    m = initialize(m)
    solver = get_solver()

    dof = degrees_of_freedom(m)
    print("dof of full model", dof)

    # solving dynamic model at steady-state
    print("solving dynamic model at steady-state...")
    solver.solve(m, tee=True)
    
    return m


def add_unit_models(m):
    """
    This function adds unit models to the steam cycle sub-flowsheet
    
    The unit models added include:

    * Feedwater Heaters
        ** FWH1 + drain mixer
        ** FWH2 + drain mixer + drain cooling
        ** FWH3 + drain mixer + drain cooling
        ** FWH5 + drain mixer + drain cooling + desuperheat
        ** FWH6 + drain mixer + drain cooling + desuperheat 

        NOTE: FWH were model using the dynamic IDAES FWH model 
                with dynamic set to 'False'     
    
    * Mixers
        ** Mixer @ FWH1 for condensate and drain 
        ** Mixer @ FWH4 deaerator (deaerator modeled as mixer)
    
    * Pumps
        ** Pump @ FWH1 for drain mix
        ** Pump @ FWH4 -- Booster pump
        ** Pump @ FWH4 -- Boiler feed water pump
    
    * Valves
        ** Valve @ FWH2, FWH3, FWH5, FWH6
            
    * Separator
        ** Separator at FWH4

    * Tank
        ** Water Tank @ FWH4 

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel

    Returns
    -------
    m : pyomo.environ.ConcreteModel
        The updated Pyomo model
    """
    fs = m.fs.stc
    prop_water = m.fs.prop_water

    # FWH1
    ##########################################################################
    # Unit model for feed water heater 1
    fs.fwh1 = FWH0D(
        dynamic=False,
        has_desuperheat=False,
        has_drain_cooling=False,
        has_drain_mixer=True,
        condense={
            "tube": {"has_pressure_change": True},
            "shell": {"has_pressure_change": True},
            "has_holdup": True,
        },
        property_package=prop_water,
    )

    # Unit model for mixer of FWH1 drain and condensate
    fs.fwh1_drain_return = Mixer(
        dynamic=False,
        inlet_list=["feedwater", "fwh1_drain"],
        property_package=prop_water,
        momentum_mixing_type=MomentumMixingType.equality,
    )

    # Drain pump for FWH1
    fs.fwh1_drain_pump = WaterPump(dynamic=False, property_package=prop_water)

    # FWH2
    ##########################################################################
    # Unit model for feed water heater 2
    fs.fwh2 = FWH0D(
        dynamic=False,
        has_desuperheat=False,
        has_drain_cooling=True,
        has_drain_mixer=True,  

        condense={
            "tube": {"has_pressure_change": True},
            "shell": {"has_pressure_change": True},
            "has_holdup": True,
        },
        property_package=prop_water,
    )

    # Unit model for water control valve between drain of fwh2 and fwh1
    fs.fwh2_valve = Valve(
        dynamic=False, 
        has_holdup=False, 
        phase="Liq", 
        property_package=prop_water
    )

    # FWH3
    ##########################################################################
    # ADD FWH3 with the valve (between FWH3 hot side return and FWH2)
    fs.fwh3 = FWH0D(
        dynamic=False,
        has_desuperheat=False,
        has_drain_cooling=True,
        has_drain_mixer=False,  

        condense={
            "tube": {"has_pressure_change": True},
            "shell": {"has_pressure_change": True},
            "has_holdup": True,
        },
        property_package=prop_water,
    )

    # Unit model for water control valve between drain of fwh2 and fwh1
    fs.fwh3_valve = Valve(
        dynamic=False, 
        has_holdup=False, 
        phase="Liq", 
        property_package=prop_water
    )

    # FWH4
    ##########################################################################
    # unit model for fwh4 deaerator
    fs.fwh4_deair = Mixer(
        dynamic=False,
        momentum_mixing_type=MomentumMixingType.equality,
        inlet_list=["steam", "drain", "feedwater"],
        property_package=prop_water,
    )

    # Unit model for deaerator water tank
    # Modeled as a horizontal cylindrical tank
    fs.da_tank = WaterTank(
        tank_type="horizontal_cylindrical_tank",
        has_holdup=True,
        property_package=prop_water,
    )

    # Unit model for electrical feedwater booster pump
    fs.booster = WaterPump(dynamic=False, property_package=prop_water)

    # Unit model for main boiler feed water pump driven by steam turbine
    fs.bfp = WaterPump(dynamic=False, property_package=prop_water)

    # Unit model for splitter for spray water stream for main attemperator
    fs.split_attemp = Separator(
        dynamic=False, property_package=prop_water, outlet_list=["FeedWater", "Spray"]
    )

    # FWH5
    ##########################################################################
    fs.fwh5 = FWH0D(
        dynamic=False,
        has_desuperheat=True,
        has_drain_cooling=True,
        has_drain_mixer=True,
        condense={
            "tube": {"has_pressure_change": True},
            "shell": {"has_pressure_change": True},
            "has_holdup": True,
        },
        desuperheat={"dynamic": False},
        cooling={"dynamic": False, "has_holdup": False},
        property_package=prop_water,
    )

    # Unit model for water control valve drain of fwh5 and deaerator
    fs.fwh5_valve = Valve(
        dynamic=False, has_holdup=False, phase="Liq", property_package=prop_water
    )

    # FWH6
    #########################################################################
    # Unit model for feed water heater 6
    fs.fwh6 = FWH0D(
        dynamic=False,
        has_desuperheat=True,
        has_drain_cooling=True,
        has_drain_mixer=False,
        condense={
            "tube": {"has_pressure_change": True},
            "shell": {"has_pressure_change": True},
            "has_holdup": True,
        },
        desuperheat={"dynamic": False},
        cooling={"dynamic": False, "has_holdup": False},
        property_package=prop_water,
    )

    # Unit model for water control valve between drain of fwh6 and fwh5
    fs.fwh6_valve = Valve(
        dynamic=False, has_holdup=False, phase="Liq", property_package=prop_water
    )

    return m


def set_arcs_and_constraints(m):
    """
    This method adds arcs to connect streams between the different 
    unit models defined in the flowsheet. 

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel

    Returns
    -------
    m : pyomo.environ.ConcreteModel
        The updated Pyomo model
    """
    fs = m.fs.stc
    # Add arcs
    # Cold side outlet to drain return mixer "feedwater" inlet 
    fs.FWH1A = Arc(
        source=fs.fwh1.condense.cold_side_outlet,
        destination=fs.fwh1_drain_return.feedwater,
    )

    # Hot side outlet to drain pump inlet
    fs.FWH1_DRN1 = Arc(
        source=fs.fwh1.condense.hot_side_outlet, 
        destination=fs.fwh1_drain_pump.inlet
    )

    # Drain pump outlet to drain return mixer "drain" inlet
    fs.FWH1_DRN2 = Arc(
        source=fs.fwh1_drain_pump.outlet, 
        destination=fs.fwh1_drain_return.fwh1_drain
    )

    # Drain return mixer outlet to fwh2 cooling cold side inlet
    fs.FWH1_FWH2_1 = Arc(
        source=fs.fwh1_drain_return.outlet, 
        destination=fs.fwh2.cooling.cold_side_inlet
    )

    # FWH2 Cooling hot side outlet to FWH2 Valve
    fs.FWH2_VLV1 = Arc(
        source=fs.fwh2.cooling.hot_side_outlet, 
        destination=fs.fwh2_valve.inlet
    )

    # FWH2 valve outlet to FWH1 hot side drain state
    fs.FWH2_VLV2 = Arc(
        source=fs.fwh2_valve.outlet, 
        destination=fs.fwh1.drain_mix.drain
    )

    # Arc between fwh3 cooling hot side outlet fwh3 valve inlet
    fs.FWH3_VLV1 = Arc(
        source=fs.fwh3.cooling.hot_side_outlet, 
        destination=fs.fwh3_valve.inlet
    )

    # Arc between fwh3 valve outlet and fwh2 condense hot side inlet drain
    fs.FWH3_VLV1_FWH2 = Arc(
        source=fs.fwh3_valve.outlet,
        destination=fs.fwh2.drain_mix.drain
    )

    # Arc between fwh2 cold side and fwh3 cooling inlet
    fs.FWH2_FWH3 = Arc(
        source=fs.fwh2.condense.cold_side_outlet,
        destination=fs.fwh3.cooling.cold_side_inlet
    )

    # Arc between fwh3 condense outlet and fwh4 dearator
    fs.FWH3_FWH4 = Arc(
        source=fs.fwh3.condense.cold_side_outlet,
        destination=fs.fwh4_deair.feedwater
    )

    # Arc between fwh4 deaerator outlet and deaerator tank
    fs.FWH4_FWH4Tank = Arc(
        source=fs.fwh4_deair.outlet,
        destination=fs.da_tank.inlet
    )

    # Arc between deaeator tank and booster inlet
    fs.FWH4Tank_FWH4Booster = Arc(
        source=fs.da_tank.outlet, 
        destination=fs.booster.inlet
    )

    # Arc between booster outlet and boiler feed water pump inlet
    fs.FWH4Booster_FWH4BFP = Arc(
        source=fs.booster.outlet, 
        destination=fs.bfp.inlet
    )

    # Arc between boiler feed water pump outlet and splitter
    fs.FWH4BFP_FWH4Split = Arc(
        source=fs.bfp.outlet, 
        destination=fs.split_attemp.inlet
    )

    # Arc between fwh5 cooling hot side outlet and fwh5 valve
    fs.FWH5_FWH5VLV = Arc(
        source = fs.fwh5.cooling.hot_side_outlet,
        destination = fs.fwh5_valve.inlet
    )

    # Arc between fwh5 valve and fwh4 deaerator
    fs.FWH5VLV_FWH4 = Arc(
        source = fs.fwh5_valve.outlet,
        destination = fs.fwh4_deair.drain
    )

    # Arc between splitter and fwh5 cooling cold side inlet
    fs.FWH4Split_FWH5 = Arc(
        source = fs.split_attemp.FeedWater,
        destination = fs.fwh5.cooling.cold_side_inlet
    )

    # Arc between fwh6 cooling hot side outlet and fwh6 valve
    fs.FWH6_FWH6VLV = Arc(
        source = fs.fwh6.cooling.hot_side_outlet,
        destination = fs.fwh6_valve.inlet
    )

    # Arc between fwh6 valve and fwh5 drain mix
    fs.FWH6VLV_FWH5 = Arc(
        source = fs.fwh6_valve.outlet,
        destination = fs.fwh5.drain_mix.drain
    )

    # Arc between fwh5 desuperheat cold side outlet and fwh6 cooling cold side inlet
    fs.FWH5_FWH6 = Arc(
        source = fs.fwh5.desuperheat.cold_side_outlet,
        destination = fs.fwh6.cooling.cold_side_inlet
    )

    # Apply the above arc connections
    pyo.TransformationFactory("network.expand_arcs").apply_to(fs)

    # Add Constraints
    @fs.Constraint(fs.time)
    def fwh1_drain_mixer_pressure_eqn(b, t):
        return (
            b.fwh1.drain_mix.drain.pressure[t] * 1e-4
            == b.fwh1.drain_mix.steam.pressure[t] * 1e-4
        )

    @fs.Constraint(fs.time)
    def fwh2_drain_mixer_pressure_eqn(b, t):
        return (
            b.fwh2.drain_mix.drain.pressure[t] * 1e-4
            == b.fwh2.drain_mix.steam.pressure[t] * 1e-4
        )

    @fs.Constraint(fs.time)
    def fwh5_drain_mixer_pressure_eqn(b, t):
        return (
            b.fwh5.drain_mix.drain.pressure[t] * 1e-5
            == b.fwh5.drain_mix.steam.pressure[t] * 1e-5
        )

    return m


def set_inputs(m):
    """
    Set unit model inputs related to dimensions, pararameters,
    and fixed design and operating variables
    Some of fixed inputs are for initialization only and will be unfixed
    before the sub-flowsheet is solved

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel

    Returns
    -------
    m : pyomo.environ.ConcreteModel
        The updated Pyomo model
    """

    fs = m.fs.stc

    # FWH1
    #-------------------------------------------------------------------------------------
    ## Condense
    fs.fwh1.condense.area.fix(1276)
    fs.fwh1.condense.overall_heat_transfer_coefficient.fix(2800)
    fs.fwh1.condense.tube.deltaP[:].fix(0)
    fs.fwh1.condense.vol_frac_shell.fix(0.75)
    fs.fwh1.condense.heater_diameter.fix(1.3)
    fs.fwh1.condense.cond_sect_length.fix(8)  
    fs.fwh1.condense.level.fix(0.275)   
    fs.fwh1.condense.tube.volume.fix(1.5)

    ## Cold Side
    ### Properties in
    Temp_cold_in = 314.9967133546581 # degrees K
    hc_in = iapws95.htpx(Temp_cold_in * pyunits.K, 694237.1088697321 * pyunits.Pa)
    fs.fwh1.condense.cold_side.properties_in[0].flow_mol.fix(19594.23812)
    fs.fwh1.condense.cold_side.properties_in[0].enth_mol.fix(hc_in)
    fs.fwh1.condense.cold_side.properties_in[0].pressure.fix(694237.1088697321) 

    ## Hot Side
    ### FWH1 drain mixer - steam from LP turbine
    fs.fwh1.drain_mix.steam_state[0].flow_mol.fix(1158.273)
    fs.fwh1.drain_mix.steam_state[0].enth_mol.fix(47733.65373991247)
    fs.fwh1.drain_mix.steam_state[0].pressure.fix(50303.553112314716) 

    ### FWH1 drain mixer - condensate from FWH2
    #### These values will be passed via arc between FWH2 drain return and FWH1 drain mixer
    #### set initial value for initialization purposes then unfix
    fs.fwh1.drain_mix.drain_state[0].flow_mol.fix(2034.622174600547)
    fs.fwh1.drain_mix.drain_state[0].enth_mol.fix(30936.596777156497)
    fs.fwh1.drain_mix.drain_state[0].pressure.fix(143724.43746375633)

    ## fwh1 drain pump
    fs.fwh1_drain_pump.deltaP[:].value = 7e5
    fs.fwh1_drain_pump.efficiency_isentropic.fix(0.8) 

    # FWH2
    #-------------------------------------------------------------------------------------
    # Condense
    fs.fwh2.condense.area.fix(1276)
    fs.fwh2.condense.overall_heat_transfer_coefficient.fix(3250)
    fs.fwh2.condense.tube.deltaP[:].fix(0) 
    fs.fwh2.condense.vol_frac_shell.fix(0.75)
    fs.fwh2.condense.heater_diameter.fix(1.3)
    fs.fwh2.condense.cond_sect_length.fix(8)  
    fs.fwh2.condense.level.fix(0.275)
    fs.fwh2.condense.tube.volume.fix(1.5)

    ## Cooling
    fs.fwh2.cooling.area.fix(154)
    fs.fwh2.cooling.overall_heat_transfer_coefficient.fix(2000)

    ## valve
    fs.fwh2_valve.Cv.value = 13.134436328638365
    fs.fwh2_valve.Cv.unfix()
    fs.fwh2_valve.valve_opening.fix(0.5)

    ## Hot Side
    ### Properties In
    #### FWH2 Condense - Condensate from FWH3
    fs.fwh2.drain_mix.drain_state[0].flow_mol.fix(941.296260461533)
    fs.fwh2.drain_mix.drain_state[0].enth_mol.fix(8165.843379740567)
    fs.fwh2.drain_mix.drain_state[0].pressure.fix(143724.43746375633) 

    #### FWH2 Condense - Steam from LP Turbine
    fs.fwh2.drain_mix.steam_state[0].flow_mol.fix(1093.3252)
    fs.fwh2.drain_mix.steam_state[0].enth_mol.fix(50541.02098744197)
    fs.fwh2.drain_mix.steam_state[0].pressure.fix(143724.43746375633)


    # FWH3
    #-------------------------------------------------------------------------------------
    ## Condense
    fs.fwh3.condense.area.fix(1430)
    fs.fwh3.condense.overall_heat_transfer_coefficient.fix(3600)
    fs.fwh3.condense.tube.deltaP[:].fix(0) 
    fs.fwh3.condense.vol_frac_shell.fix(0.7)
    fs.fwh3.condense.heater_diameter.fix(1.3)
    fs.fwh3.condense.cond_sect_length.fix(7.5)  
    fs.fwh3.condense.level.fix(0.275)
    fs.fwh3.condense.tube.volume.fix(1.5)

    # Cooling
    fs.fwh3.cooling.area.fix(145)
    fs.fwh3.cooling.overall_heat_transfer_coefficient.fix(1850)

    # valve
    fs.fwh3_valve.Cv.value = 5
    fs.fwh3_valve.Cv.unfix()
    fs.fwh3_valve.valve_opening.fix(0.48271675)
    fs.fwh3_valve.control_volume.properties_out[0].pressure[:]=143724.43746375633
    fs.fwh3_valve.control_volume.properties_out[0].pressure.unfix()

    # Condense hot side inlet
    fs.fwh3.condense.hot_side.properties_in[0].flow_mol[:] = 941.29
    fs.fwh3.condense.hot_side.properties_in[0].flow_mol.unfix()
    fs.fwh3.condense.hot_side.properties_in[0].enth_mol.fix(52896.52208800597) 
    fs.fwh3.condense.hot_side.properties_in[0].pressure.fix(293315.1784974619)

    # FWH4 deaerator
    #-------------------------------------------------------------------------------------
    ## input from turbine
    fs.fwh4_deair.steam.flow_mol.fix(561.3418581986799*2)
    fs.fwh4_deair.steam.enth_mol.fix(56264.11300967524)
    fs.fwh4_deair.steam.pressure[0].set_value(694237.1088697321)
    fs.fwh4_deair.steam.pressure.unfix()

    ## input from fwh5 (Temporary - to be unfixed and passed when fwh5 is added)
    fs.fwh4_deair.drain.flow_mol.fix(1750.9034119867788*2)
    fs.fwh4_deair.drain.enth_mol.fix(13097.807067762913)
    fs.fwh4_deair.drain.pressure.fix(694237.1088697321)

    # Set inputs for deaerator tank
    fs.da_tank.tank_diameter.fix(4.0)
    fs.da_tank.tank_length.fix(20)
    fs.da_tank.tank_level.fix(2.75)

    # Set inputs for booster pump
    fs.booster.efficiency_isentropic.fix(0.8)
    fs.booster.outlet.pressure.fix(1.5e6)

    # Set inputs for main boiler feed water pump
    fs.bfp.efficiency_isentropic.fix(0.8)
    fs.bfp.outlet.pressure.fix(1.45e7)

    # Set input for splitter for main steam attemperator spray
    fs.split_attemp.split_fraction[:, "Spray"].fix(0.0007)


    # FWH5
    #-------------------------------------------------------------------------------------
    # set inputs for cooling
    fs.fwh5.cooling.area.fix(220) #original areax2.2
    fs.fwh5.cooling.overall_heat_transfer_coefficient.fix(2000)

    # set inputs for condense
    fs.fwh5.condense.area.fix(1320) #original areax2.2
    fs.fwh5.condense.overall_heat_transfer_coefficient.fix(3050)
    fs.fwh5.condense.tube.deltaP[:].fix(0) 
    fs.fwh5.condense.vol_frac_shell.fix(0.75)
    fs.fwh5.condense.heater_diameter.fix(1.3)
    fs.fwh5.condense.cond_sect_length.fix(8)  
    fs.fwh5.condense.level.fix(0.275)
    fs.fwh5.condense.tube.volume.fix(1.5)

    # set inputs for desuperheat
    fs.fwh5.desuperheat.area.fix(187) #original areax2.2
    fs.fwh5.desuperheat.overall_heat_transfer_coefficient.fix(450)

    # set inputs for valve; exclude until we initialize and solve fwh5
    # Set inputs for level control valve after FWH5
    fs.fwh5_valve.Cv.value = 5.156
    fs.fwh5_valve.valve_opening.fix(0.5)

    # desuperheat hot side inlet from turbine
    fs.fwh5.desuperheat.hot_side.properties_in[0].flow_mol.fix(434.94041786171243*2.2)
    fs.fwh5.desuperheat.hot_side.properties_in[0].enth_mol.fix(59084.67377032232)
    fs.fwh5.desuperheat.hot_side.properties_in[0].pressure.fix(1214153.3628251995)

    # condense hot side inlet from fwh6 (temporary - to be adjusted when fwh6 is added)
    fs.fwh5.drain_mix.drain_state[0].flow_mol.fix(1315.9629941250662*2.2)
    fs.fwh5.drain_mix.drain_state[0].enth_mol.fix(14973.540670736045)
    fs.fwh5.drain_mix.drain_state[0].pressure.fix(1214153.3628251995)


    # FWH6
    #-------------------------------------------------------------------------------------
    # set inputs for cooling
    fs.fwh6.cooling.area.fix(286) #original areax2.2
    fs.fwh6.cooling.overall_heat_transfer_coefficient.fix(1900)

    # set inputs for condense
    fs.fwh6.condense.area.fix(1650) #original areax2.2
    fs.fwh6.condense.overall_heat_transfer_coefficient.fix(3200)
    fs.fwh6.condense.tube.deltaP[:].fix(0) 
    fs.fwh6.condense.vol_frac_shell.fix(0.75)
    fs.fwh6.condense.heater_diameter.fix(1.3)
    fs.fwh6.condense.cond_sect_length.fix(8)  
    fs.fwh6.condense.level.fix(0.275)
    fs.fwh6.condense.tube.volume.fix(1.5)

    # set inputs for desuperheat
    fs.fwh6.desuperheat.area.fix(275) #original areax2.2
    fs.fwh6.desuperheat.overall_heat_transfer_coefficient.fix(780)

    # Set inputs for level control valve after FWH6
    fs.fwh6_valve.Cv.value = 1.9986
    fs.fwh6_valve.valve_opening.fix(0.5)

    # desuperheat hot side inlet from turbine
    fs.fwh6.desuperheat.hot_side.properties_in[0].flow_mol.fix(1315.9629941250662*2.2)
    fs.fwh6.desuperheat.hot_side.properties_in[0].enth_mol.fix(55812.12757170066)
    fs.fwh6.desuperheat.hot_side.properties_in[0].pressure.fix(3438698.4168059407)

    return m


def initialize(m):

    """
    This method initializes the model 

    Notes
    -----
    * Several fixed variables are unfixed after initialization.
    * The propagate_state method is used to pass outlet states to subsequent
      unit models between initialization steps.
    * 06/25/2025 - The main unfixed variables left out before solving include:
        ** Intermediate outlet states/flows passed between connected units
           (e.g., flow from FWH1 to FWH2). These are initialized with
           propagate_state.
        ** Steam extraction flows from the turbines to selected feedwater
           heaters. These are initialized based on guesses (2.2xintial flow), 
           then unfixed so the full steady-state flowsheet can solve for the
           extraction flow.
        ** Valve coefficients for selected drain/extraction valves. 
           These may be solved so the pressure drop and flow
           through the connected heaters are consistent.
    * Feedwater boundary conditions, heater area, and heat-transfer
      coefficients are kept fixed where they represent design or operating
      specifications.
    
    Parameters
    ----------
    m : pyomo.environ.ConcreteModel

    Returns
    -------
    m : pyomo.environ.ConcreteModel
        The updated Pyomo model
    """
    
    fs = m.fs.stc
    outlvl = idaeslog.DEBUG
    solver = get_solver()

    # Initialize fwh1
    fs.fwh1.initialize(outlvl=outlvl, optarg=solver.options)

    # When solving the model, the U, A, and input flow will remain fixed
    # The steam flow is unfixed to allow the model to solve for the 
    # required steam extraction flow
    fs.fwh1.drain_mix.steam.flow_mol[0].unfix()

    # unfix FWH1 output conditions
    # these will be calculated when solving and passed to the next FWH
    fs.fwh1.condense.cold_side.properties_out[0.0].pressure.unfix()

    # propagate fwh1 cold side output to the drain return mixer inlet
    _set_port(fs.fwh1_drain_return.feedwater, 
                fs.fwh1.condense.cold_side_outlet
    )

    # propagate fwh1 hot side output to the drain pump
    _set_port(fs.fwh1_drain_pump.inlet, 
                fs.fwh1.condense.hot_side_outlet
    )

    # Temporarily set the drain pump outlet pressure to the feedwater-line 
    # pressure - unfix after intialization
    fs.fwh1_drain_pump.outlet.pressure[0].fix(
        pyo.value(fs.fwh1.condense.cold_side_outlet.pressure[0])
    )

    # initialize fwh1 drain pump
    fs.fwh1_drain_pump.initialize(outlvl=outlvl, optarg=solver.options)

    # unfix pump outlet pressure
    fs.fwh1_drain_pump.outlet.pressure[0].unfix()

    # propogate drain pump outlet to drain return mixer inlet
    _set_port(fs.fwh1_drain_return.fwh1_drain, 
                fs.fwh1_drain_pump.outlet
    )

    # initiate drain return mixer
    fs.fwh1_drain_return.initialize(outlvl=outlvl, optarg=solver.options)

    # FWH1 inputs from FWH2
    fs.fwh1.drain_mix.drain_state[0.0].flow_mol.unfix()
    fs.fwh1.drain_mix.drain_state[0.0].enth_mol.unfix()
    fs.fwh1.drain_mix.drain_state[0].pressure.unfix()

    # Propagate fwh1 results to fwh2 cold side inlet
    _set_port(fs.fwh2.cooling.cold_side_inlet,
            fs.fwh1_drain_return.outlet
    )

    # Temporarily fix the complete propagated FWH1 outlet state so FWH2 has
    # zero degrees of freedom during unit initialization. 
    # These variables are then unfixed so that the values are passed when 
    # solving based on the set arc connexting fwh1 and fwh2.
    fs.fwh2.cooling.cold_side.properties_in[0].flow_mol.fix()
    fs.fwh2.cooling.cold_side.properties_in[0].enth_mol.fix()
    fs.fwh2.cooling.cold_side.properties_in[0].pressure.fix()

    # initialize FWH2
    fs.fwh2.initialize(outlvl=outlvl, optarg=solver.options)

    # Similar to FWH1
    # When solving the model, the U, A, and input flow will remain fixed
    # The steam flow is unfixed to allow the model to solve for the 
    # required steam extraction flow for FWH2
    fs.fwh2.drain_mix.steam.flow_mol[0].unfix()

    # propagate results from the fwh2 cooling hot side outlet to FWH2 valve
    _set_port(fs.fwh2_valve.inlet,
            fs.fwh2.cooling.hot_side_outlet
    )

    # Initialize against the known FWH1 shell pressure and calculate a
    # consistent Cv. 
    fs.fwh2_valve.outlet.pressure[0].fix(
        pyo.value(fs.fwh1.drain_mix.steam.pressure[0])
    )

    # intialize fwh2 valve
    fs.fwh2_valve.initialize(
        outlvl=outlvl, optarg=solver.options, calculate_cv=True
    )

    # unfix pump outlet pressure
    fs.fwh2_valve.outlet.pressure[0].unfix()

    # Propagate the fwh2 valve outlet to the the fwh1 drain mixer 
    _set_port(fs.fwh1.drain_mix.drain, 
            fs.fwh2_valve.outlet
    )

    # Initialize fwh1 drain mix
    fs.fwh1.drain_mix.initialize(outlvl=outlvl, optarg=solver.options)

    # propagate fwh1 drain mix outlet to the hot side outlet
    _set_port(fs.fwh1.condense.hot_side_inlet, 
            fs.fwh1.drain_mix.outlet
    )

    # unfix the fwh2 input from fwh1 (when solving this would be
    # passed via an arc)
    fs.fwh2.cooling.cold_side.properties_in[0].flow_mol.unfix() 
    fs.fwh2.cooling.cold_side.properties_in[0].enth_mol.unfix()
    fs.fwh2.cooling.cold_side.properties_in[0].pressure.unfix()

    # unfix the fwh2 input from fwh3 (when solving this would be
    # passed via an arc)
    fs.fwh2.drain_mix.drain_state[0].flow_mol.unfix()
    fs.fwh2.drain_mix.drain_state[0].enth_mol.unfix()
    fs.fwh2.drain_mix.drain_state[0].pressure.unfix() 

    # Propagate fwh2 results to fwh3 cold side inlet
    _set_port(fs.fwh3.cooling.cold_side_inlet,
            fs.fwh2.condense.cold_side_outlet
    )

    # temporarily fix fwh3 inlet conditions (for initialization)
    fs.fwh3.cooling.cold_side.properties_in[0].flow_mol.fix()
    fs.fwh3.cooling.cold_side.properties_in[0].enth_mol.fix()
    fs.fwh3.cooling.cold_side.properties_in[0].pressure.fix()

    # initialize fwh3
    fs.fwh3.initialize(outlvl=outlvl, optarg=solver.options)

    # unfix the fwh3 input from fwh2 (when solving this would be
    # passed via an arc)
    fs.fwh3.cooling.cold_side.properties_in[0].flow_mol.unfix()
    fs.fwh3.cooling.cold_side.properties_in[0].enth_mol.unfix()
    fs.fwh3.cooling.cold_side.properties_in[0].pressure.unfix()

    # propagate the fwh3 cooling outlet result to the valve inlet
    _set_port(fs.fwh3_valve.inlet,
            fs.fwh3.cooling.hot_side_outlet
    )

    # Initialize the fwh3 drain valve at the using the fwh2 hot side pressure
    # calculate Cv for the specified opening and initialized drain flow.
    fs.fwh3_valve.outlet.pressure[0].fix(
        pyo.value(fs.fwh2.drain_mix.steam.pressure[0])
    )

    # intialize fwh3 valve
    fs.fwh3_valve.initialize(
        outlvl=outlvl, optarg=solver.options, calculate_cv=True
    )

    # unfix the pressure - this would be an output when solving the full model
    fs.fwh3_valve.outlet.pressure[0].unfix()

    # propagate valve outlet results to the fwh2 drain mix
    _set_port(fs.fwh2.drain_mix.drain, 
            fs.fwh3_valve.outlet
    )

    # propagate fwh3 condense cold side outlet results to fwh4 deaerator feedwater
    _set_port(fs.fwh4_deair.feedwater,
            fs.fwh3.condense.cold_side_outlet
    )

    # initialize fwh4 deaerator
    fs.fwh4_deair.initialize(
        outlvl=outlvl, optarg=solver.options
    )

    # propagate fwh4 deaerator output to deair_tank inlet
    _set_port(fs.da_tank.inlet,
            fs.fwh4_deair.outlet
    )

    # initialize the deaerator tank
    fs.da_tank.initialize(
        outlvl = outlvl, 
        optarg = solver.options
    )

    # propagate deaerator tank to booster pump
    _set_port(fs.booster.inlet,
            fs.da_tank.outlet
    )

    # initialize booster pump
    fs.booster.initialize(
        outlvl = outlvl, 
        optarg = solver.options
    )

    # propagate booster pump to bfp
    _set_port(fs.bfp.inlet,
            fs.booster.outlet
    )

    # initialize bfp
    fs.bfp.initialize(
        outlvl = outlvl, 
        optarg = solver.options
    )

    # propagate bfp to splitter
    _set_port(fs.split_attemp.inlet,
            fs.bfp.outlet
    )

    # intialize splitter
    fs.split_attemp.initialize(
        outlvl=outlvl, 
        optarg=solver.options
    )

    # propagate splitter output to fwh5 cooling inlet
    _set_port(
        fs.fwh5.cooling.cold_side_inlet,
        fs.split_attemp.FeedWater
    )

    # temporarily fix fwh5 inlet conditions (for initialization)
    fs.fwh5.cooling.cold_side.properties_in[0].flow_mol.fix()
    fs.fwh5.cooling.cold_side.properties_in[0].enth_mol.fix()
    fs.fwh5.cooling.cold_side.properties_in[0].pressure.fix()

    # initialize fwh5
    fs.fwh5.initialize(
        outlvl=outlvl, 
        optarg=solver.options
    )

    # similar to fwh1 and fwh2
    # Fixed FWH5 area, U, and incoming drain and feedwater 
    # Unfixing the stream extration flow (to be solved).
    fs.fwh5.desuperheat.hot_side.properties_in[0].flow_mol.unfix()

    # unfix the fwh5 input from fwh4 (when solving this would be
    # passed via an arc)
    fs.fwh5.cooling.cold_side.properties_in[0].flow_mol.unfix()
    fs.fwh5.cooling.cold_side.properties_in[0].enth_mol.unfix()
    fs.fwh5.cooling.cold_side.properties_in[0].pressure.unfix()

    # propagate fwh5 output to fwh5 valve
    _set_port(fs.fwh5_valve.inlet,
            fs.fwh5.cooling.hot_side_outlet
    )

    # Initialize the FWH5 drain valve at deaerator pressure and calculate a
    # consistent Cv for the specified opening and initialized drain flow.
    fs.fwh5_valve.outlet.pressure[0].fix(
        pyo.value(fs.fwh4_deair.steam.pressure[0])
    )
    fs.fwh5_valve.initialize(
        outlvl=outlvl, optarg=solver.options, calculate_cv=True
    )
    fs.fwh5_valve.outlet.pressure[0].unfix()

    # propagate fwh5 valve outlet to fwh4 hot_side input
    _set_port(fs.fwh4_deair.drain, 
            fs.fwh5_valve.outlet
    )

    # unfix fwh4 deair drain input
    fs.fwh4_deair.drain.flow_mol.unfix()
    fs.fwh4_deair.drain.enth_mol.unfix()
    fs.fwh4_deair.drain.pressure.unfix()

    # propagate splitter output to fwh6 cooling inlet
    _set_port(
        fs.fwh6.cooling.cold_side_inlet,
        fs.fwh5.desuperheat.cold_side_outlet
    )

    # temporarily fix fwh6 inlet conditions (for initialization)
    fs.fwh6.cooling.cold_side.properties_in[0].flow_mol.fix()
    fs.fwh6.cooling.cold_side.properties_in[0].enth_mol.fix()
    fs.fwh6.cooling.cold_side.properties_in[0].pressure.fix()

    # initialize fwh6
    fs.fwh6.initialize(
        outlvl=outlvl, 
        optarg=solver.options
    )

    # similar to FWH 1, 2, and 5
    # Fixed FWH6 area, U, and incoming drain and feedwater 
    # Unfixing the stream extration flow (to be solved).
    fs.fwh6.desuperheat.hot_side.properties_in[0].flow_mol.unfix()

    # unfix the fwh6 input from fwh5 (when solving this would be
    # passed via an arc)
    fs.fwh6.cooling.cold_side.properties_in[0].flow_mol.unfix()
    fs.fwh6.cooling.cold_side.properties_in[0].enth_mol.unfix()
    fs.fwh6.cooling.cold_side.properties_in[0].pressure.unfix()

    # propagate fwh6 output to fwh6 valve
    _set_port(fs.fwh6_valve.inlet,
            fs.fwh6.cooling.hot_side_outlet
    )

    # Initialize the FWH6 drain valve at FWH5 pressure and calculate a
    # consistent Cv for the specified opening and initialized drain flow.
    fs.fwh6_valve.outlet.pressure[0].fix(
        pyo.value(fs.fwh5.drain_mix.steam.pressure[0])
    )
    fs.fwh6_valve.initialize(
        outlvl=outlvl, optarg=solver.options, calculate_cv=True
    )
    fs.fwh6_valve.outlet.pressure[0].unfix()

    # Propagate FWH6 drain-valve outlet to FWH5's drain-mixer inlet.
    _set_port(fs.fwh5.drain_mix.drain, 
            fs.fwh6_valve.outlet
    )
    fs.fwh5.drain_mix.drain.flow_mol.unfix()
    fs.fwh5.drain_mix.drain.enth_mol.unfix()
    fs.fwh5.drain_mix.drain.pressure.unfix()

    # Reinitialize the drain_mix to capture the updated flows
    fs.fwh1.drain_mix.initialize(
        outlvl=outlvl, 
        optarg=solver.options
    )

    fs.fwh2.drain_mix.initialize(
        outlvl=outlvl, 
        optarg=solver.options
    )

    fs.fwh5.drain_mix.initialize(
        outlvl=outlvl, 
        optarg=solver.options
    )

    return m


def set_scaling_factors(m):
    """Set scaling factors for variables and expressions. These are used for
    variable scaling and used by the framework to scale constraints.

    Args:
        m: plant model to set scaling factors for.

    Returns:
        None
    """
    fs = m.fs.stc

    # Set Scaling
    ## fwh1
    iscale.set_scaling_factor(fs.fwh1.condense.hot_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh1.condense.hot_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh1.condense.cold_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh1.condense.cold_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh1.condense.hot_side.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh1.condense.cold_side.heat, 1e-7)

    ## fwh2
    iscale.set_scaling_factor(fs.fwh2.condense.hot_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh2.condense.hot_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh2.condense.cold_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh2.condense.cold_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh2.condense.hot_side.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh2.condense.cold_side.heat, 1e-7)

    ## fwh3
    iscale.set_scaling_factor(fs.fwh3.condense.hot_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh3.condense.hot_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh3.condense.cold_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh3.condense.cold_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh3.condense.hot_side.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh3.condense.cold_side.heat, 1e-7)

    # deaerator tank
    iscale.set_scaling_factor(fs.da_tank.control_volume.energy_holdup, 1e-10)
    iscale.set_scaling_factor(fs.da_tank.control_volume.material_holdup, 1e-6)

    # fwh5
    iscale.set_scaling_factor(fs.fwh5.condense.hot_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh5.condense.hot_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh5.condense.cold_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh5.condense.cold_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh5.condense.hot_side.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh5.condense.cold_side.heat, 1e-7)

    # fwh6
    iscale.set_scaling_factor(fs.fwh6.condense.hot_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh6.condense.hot_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh6.condense.cold_side.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh6.condense.cold_side.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh6.condense.hot_side.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh6.condense.cold_side.heat, 1e-7)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


if __name__ == "__main__":
    # This method builds and runs a steam cycle flowsheet, the flowsheet
    # includes the Turbine train, Condenser, Feed Water Heaters and Pumps,
    # fixed inlets are steam flowrates from the boiler (Main Steam and
    # Hot Reheat) and makeup of water.
    m = main_steady_state()
