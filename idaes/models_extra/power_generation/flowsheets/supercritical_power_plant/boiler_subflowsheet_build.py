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
# =============================================================================
"""
SCPC Boiler subflowsheet

Inputs:
    BFW - boiler feed water (from Feed water heaters)
    Coal from pulverizers

Main Assumptions:
    Coal flowrate as a function of load, coal HHV is fixed and heat dutty
    split from fire side to water wall and platen superheater is fixed.

    Boiler heat exchanger network:
        Water Flow:
            BFW -> ECONOMIZER -> Water Wall -> Primary SH -> Platen SH -> Finishing 
            Superheate -> HP Turbine -> Reheater -> IP Turbine
        Flue Gas Flow:
            Fire Ball -> Platen SH -> Finishing SH -> Reheater  -> o -> Economizer -> 
            Air Preheater -> Primary SH --^

        * HP Turbine, IP Turbine, Air Preheater ==> not included in this release

    Models used:
        - Mixers: Attemperator, Flue gas mix
        - Heater: Platen SH, Fire/Water side (simplified model)
        - BoilerHeatExchanger: Economizer, Primary SH, Finishing SH, Reheater
            + Shell and tube heat exchanger
                - tube side: Steam (side 1 holdup)
                - shell side: flue gas (side 2 holdup)

    Property packages used:
        - IAPWS: Water/steam side
        - IDEAL GAS: Flue Gas side

    Numerical scaling approach
        - Scaling is done by initializing the coupled flowsheet sequentially with a 
        reconciliation pass, applying each unit’s default scaler, and using AutoScaler 
        to fill remaining gaps based on initialized variable magnitudes and Jacobian 
        row norms before the final solve.

Created: 1/10/2020 by Boiler subsystem team (M Zamarripa)
Modified: 10/01/2026 (F Hanna)
"""

# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

__author__ = "Miguel Zamarripa"

# import dependencies
from collections import OrderedDict
import os
import logging

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel,
    value,
    TransformationFactory,
    units as pyunits,
)
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.opt import check_optimal_termination

from idaes.core.util.tags import svg_tag

# Import IDAES core
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state as _set_port
from idaes.core.solvers import get_solver, AutoScaler, set_scaling_factor
from idaes.core.scaling.util import (
    get_jacobian, get_scaling_factor,
    list_unscaled_constraints, list_unscaled_variables,
)
# Import Unit Model Modules
from idaes.models.properties import iapws95

# Import Property Modules
from idaes.models_extra.power_generation.properties import FlueGasParameterBlock

# Import Unit Model Modules
from idaes.models.unit_models import Heater, Mixer
from idaes.models_extra.power_generation.unit_models.boiler_heat_exchanger import (
    BoilerHeatExchanger,
    TubeArrangement,
    HeatExchangerFlowPattern,
)
from idaes.models.unit_models.separator import (
    Separator,
    SplittingType,
    EnergySplittingType,
)


def main():
    """
    Make the flowsheet object, fix some variables, and solve the problem
    """
    # Create a Concrete Model as the top level object
    m = ConcreteModel()

    # Add a flowsheet object to the model
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.prop_water = iapws95.Iapws95ParameterBlock()

    build_boiler(m.fs)

    # Create a solver
    solver = get_solver()
    return (m, solver)

def boiler_hx(phase, radiation):
    """
    Helper function to create a boiler heat exchanger with the given phase and radiation flag.

    Parameters
    ----------
    phase : str
        The phase of the cold side water ('Liq' or 'Vap').
    radiation : bool
        Whether to include radiation in the heat exchanger.
    """
    return BoilerHeatExchanger(
        cold_side={"property_package": m.fs.prop_water, "has_pressure_change": True},
        hot_side={"property_package": m.fs.prop_fluegas, "has_pressure_change": True},
        has_holdup=False,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
        tube_arrangement=TubeArrangement.inLine,
        cold_side_water_phase=phase,
        has_radiation=radiation,
    )

def build_boiler(fs):
    """
    This function builds the boiler subflowsheet within the given flowsheet block.
    This includes creating the necessary unit models and connecting them with arcs to
    represent the flow of water/steam and flue gas through the boiler system.
    The boiler subflowsheet includes two main flow paths: 
    1. The water/steam flow path, which consists of the following components:
        * Economizer
        * Water wall
        * Primary superheater
        * Platen superheater
        * Finishing superheater
        * Reheater
    2. The flue gas flow path, which consists of the following components:
        * Finishing superheater
        * Splitter
        * Reheater
        * Mixer
    
    Parameters:
    ----------
    fs : FlowsheetBlock
        The flowsheet block to which the boiler subflowsheet will be added.
    
    Returns:
    -------
    None
    """
    # Add property packages to flowsheet library
    fs.prop_fluegas = FlueGasParameterBlock()

    # Create unit models
    # Boiler Economizer
    fs.ECON = boiler_hx("Liq", False)
    # Boiler Water Wall
    fs.Water_wall = Heater(property_package=fs.prop_water)
    # Primary Superheater
    fs.PrSH = boiler_hx("Vap", True)
    # Platen Superheater
    fs.PlSH = Heater(property_package=fs.prop_water)
    # Finishing Superheater
    fs.FSH = boiler_hx("Vap", True)
    # Reheater
    fs.RH = boiler_hx("Vap", True)
    # Boiler Splitter (splits FSH flue gas outlet to Reheater and PrSH)
    fs.Spl1 = Separator(
        property_package=fs.prop_fluegas,
        split_basis=SplittingType.totalFlow,
        energy_split_basis=EnergySplittingType.equal_temperature,
    )
    # Flue gas mixer (mixing FG from Reheater and Primary SH, inlet to ECON)
    fs.mix1 = Mixer(
        property_package=fs.prop_fluegas,
        inlet_list=["Reheat_out", "PrSH_out"],
        dynamic=False,
    )
    # Mixer for Attemperator #1 (between PrSH and PlSH)
    fs.ATMP1 = Mixer(
        property_package=fs.prop_water,
        inlet_list=["Steam", "SprayWater"],
        dynamic=False,
    )

    # Build connections (streams)
    # Steam Route (side 1 = tube side = steam/water side)
    # Boiler feed water to Economizer (to be imported in full plant model)
    #    fs.bfw2econ = Arc (source=fs.FWH8.outlet,
    #                       destination=fs.ECON.cold_side_inlet)
    fs.econ2ww = Arc(source=fs.ECON.cold_side_outlet, destination=fs.Water_wall.inlet)
    fs.ww2prsh = Arc(source=fs.Water_wall.outlet, destination=fs.PrSH.cold_side_inlet)
    fs.prsh2plsh = Arc(source=fs.PrSH.cold_side_outlet, destination=fs.PlSH.inlet)
    fs.plsh2fsh = Arc(source=fs.PlSH.outlet, destination=fs.FSH.cold_side_inlet)
    fs.FSHtoATMP1 = Arc(source=fs.FSH.cold_side_outlet, destination=fs.ATMP1.Steam)
    # The attemperator outlet is out of scope for the boiler subflowsheet
    # it will be connected to the HP turbine inlet in the full plant model
    #    fs.fsh2hpturbine = Arc(source=fs.ATMP1.outlet,
    #                           destination=fs.HPTinlet)

    # Flue gas route 
    # water wall connected with boiler block (to fix the heat duty)
    # platen SH connected with boiler block (to fix the heat duty)
    # Finishing superheater connected with a flowsheet level constraint
    fs.fg_fsh2_separator = Arc(source=fs.FSH.hot_side_outlet, destination=fs.Spl1.inlet)
    fs.fg_fsh2rh = Arc(source=fs.Spl1.outlet_1, destination=fs.RH.hot_side_inlet)
    fs.fg_fsh2PrSH = Arc(source=fs.Spl1.outlet_2, destination=fs.PrSH.hot_side_inlet)
    fs.fg_rhtomix = Arc(source=fs.RH.hot_side_outlet, destination=fs.mix1.Reheat_out)
    fs.fg_prsh2mix = Arc(source=fs.PrSH.hot_side_outlet, destination=fs.mix1.PrSH_out)
    fs.fg_mix2econ = Arc(source=fs.mix1.outlet, destination=fs.ECON.hot_side_inlet)

    TransformationFactory("network.expand_arcs").apply_to(fs)

def autoscale_unit(unit, label):
    """
    Apply the unit model's default scaler, then use AutoScaler to fill any
    remaining variable and constraint scaling gaps based on initialized
    variable magnitudes and Jacobian row norms.

    Parameters
    ----------
    unit : ProcessBlockData
        Initialized IDAES unit model to scale.
    label : str
        Descriptive unit name used in the scaling summary output.

    Returns
    -------
    tuple
        A two-element tuple containing:

        default_scaler
            The unit model's default scaler instance, or ``None`` if the unit
            does not provide one.
        gap_scaler : AutoScaler
            The AutoScaler instance used to fill missing scaling factors.    
    """
    # The Helmholtz/IAPWS default scaler requires representative molar-flow
    # factors. Supply these from initialized nominal values before invoking it.
    required_flow_factors = 0
    for variable in unit.component_data_objects(pyo.Var, descend_into=True):
        if variable.parent_component().local_name == "flow_mol":
            nominal = abs(pyo.value(variable))
            if nominal > 1e-12 and get_scaling_factor(variable) is None:
                set_scaling_factor(variable, 1 / nominal, overwrite=False)
                required_flow_factors += 1

    default_scaler = unit.default_scaler(overwrite=False)
    if default_scaler is not None:
        default_scaler.scale_model(unit)

    variables_after_default = list_unscaled_variables(
        unit, descend_into=True, include_fixed=False
    )
    constraints_after_default = list_unscaled_constraints(
        unit, descend_into=True
    )

    # Fill only gaps left by the unit's default scaler. Existing factors are
    # protected because overwrite=False.
    gap_scaler = AutoScaler(overwrite=False)
    if variables_after_default:
        gap_scaler.scale_variables_by_magnitude(unit, descend_into=True)
    if constraints_after_default:
        gap_scaler.scale_constraints_by_jacobian_norm(
            unit, norm=2, descend_into=True
        )

    remaining_variables = list_unscaled_variables(
        unit, descend_into=True, include_fixed=False
    )
    remaining_constraints = list_unscaled_constraints(
        unit, descend_into=True
    )
    scaler_name = type(default_scaler).__name__ if default_scaler is not None else "None"
    print(
        f"{label}: default={scaler_name}, required flow factors={required_flow_factors}, "
        f"gaps before AutoScaler=({len(variables_after_default)} vars, "
        f"{len(constraints_after_default)} cons), remaining="
        f"({len(remaining_variables)} vars, {len(remaining_constraints)} cons)"
    )
    return default_scaler, gap_scaler

def initialize(m):
    """

    """
    # ------------- ECONOMIZER -----------------------------------------------
    # BFW Boiler Feed Water inlet temeperature = 555 F = 563.706 K
    m.fs.ECON.cold_side_inlet.flow_mol[0].fix(24194.177)  # mol/s
    m.fs.ECON.cold_side_inlet.enth_mol[0].fix(14990.97)
    m.fs.ECON.cold_side_inlet.pressure[0].fix(26922222.222)  # Pa

    # FLUE GAS Inlet from Primary Superheater
    FGrate = 21290.6999  # mol/s
    comp={"H2O":.0869,"CO2":.1449,"N2":.7434,"O2":.0247,"NO":.0006,"SO2":.002}
    for j,x in comp.items(): 
        m.fs.ECON.hot_side_inlet.flow_mol_comp[0,j].fix(FGrate * x)
    m.fs.ECON.hot_side_inlet.temperature[0].fix(682.335)  # K
    m.fs.ECON.hot_side_inlet.pressure[0].fix(100145)  # Pa

    # Economizer design variables and parameters
    ITM = 0.0254  # inch to meter conversion
    # Based on NETL Baseline Report Rev3
    # calc inner diameter (2 = outer diameter, thickness = 0.188)
    m.fs.ECON.tube_di.fix((2 - 2 * 0.188) * ITM)                 
    m.fs.ECON.tube_thickness.fix(0.188 * ITM)  # tube thickness
    m.fs.ECON.pitch_x.fix(3.5 * ITM)
    # pitch_y = (54.5) gas path transverse width /columns
    m.fs.ECON.pitch_y.fix(5.03 * ITM)
    m.fs.ECON.tube_length.fix(53.41 * 12 * ITM)  # use tube length (53.41 ft)
    m.fs.ECON.tube_nrow.fix(36 * 2.5)  # use to match baseline performance
    m.fs.ECON.tube_ncol.fix(130)  # 130 from NETL
    m.fs.ECON.nrow_inlet.fix(2)
    m.fs.ECON.delta_elevation.fix(50)
    # parameters
    # heat transfer resistance due to tube side fouling (water scales)
    m.fs.ECON.tube_r_fouling = 0.000176
    # heat transfer resistance due to tube shell fouling (ash deposition)
    m.fs.ECON.shell_r_fouling = 0.00088
    if m.fs.ECON.config.has_radiation is True:
        m.fs.ECON.emissivity_wall.fix(0.7)  # wall emissivity
    # correction factor for overall heat transfer coefficient
    m.fs.ECON.fcorrection_htc.fix(1.5)
    # correction factor for pressure drop calc tube side
    m.fs.ECON.fcorrection_dp_tube.fix(1.0)
    # correction factor for pressure drop calc shell side
    m.fs.ECON.fcorrection_dp_shell.fix(1.0)

    # initialize the economizer unit
    m.fs.ECON.initialize(outlvl=logging.INFO)
    econ_scaler=autoscale_unit(m.fs.ECON,"ECON")

    # ------- Water wall Superheater ----------------------------------------
    # propagate the economizer outlet state to the water-wall inlet
    _set_port(arc=m.fs.econ2ww)
    m.fs.Water_wall.heat_duty[:].fix(7.51e8)  # 8.76e8

    # Intialize the water wall unit
    m.fs.Water_wall.initialize(outlvl=logging.INFO)
    ww_scaler = autoscale_unit(m.fs.Water_wall, "Water_wall")

    # --------- Primary Superheater -----------------------------------------
    # Steam from water wall
    # propagate the water-wall outlet state to the PrSH cold-side inlet
    _set_port(arc=m.fs.ww_to_prsh)
   
    # FLUE GAS Inlet from Primary Superheater
    FGrate = 21290.6999 * 0.18  # mol/s
    # Use FG molar composition to set component flow rates (baseline report)
    for j,x in comp.items(): 
        m.fs.PrSH.hot_side_inlet.flow_mol_comp[0,j].fix(FGrate*x)
    m.fs.PrSH.hot_side_inlet.temperature[0].fix(1180.335)
    m.fs.PrSH.hot_side_inlet.pressure[0].fix(100145)

    # Primary Superheater
    ITM = 0.0254  # inch to meter conversion
    m.fs.PrSH.tube_di.fix((2.5 - 2 * 0.165) * ITM)
    m.fs.PrSH.tube_thickness.fix(0.165 * ITM)
    m.fs.PrSH.pitch_x.fix(3 * ITM)
    # gas path transverse width 54.78 ft / number of columns
    m.fs.PrSH.pitch_y.fix(54.78 / 108 * 12 * ITM)
    m.fs.PrSH.tube_length.fix(53.13 * 12 * ITM)
    m.fs.PrSH.tube_nrow.fix(20 * 2)
    m.fs.PrSH.tube_ncol.fix(108)
    m.fs.PrSH.nrow_inlet.fix(4)
    m.fs.PrSH.delta_elevation.fix(50)
    m.fs.PrSH.tube_r_fouling = 0.000176  # (0.001 h-ft^2-F/BTU)
    m.fs.PrSH.shell_r_fouling = 0.003131  # (0.03131 - 0.1779 h-ft^2-F/BTU)
    if m.fs.PrSH.config.has_radiation is True:
        m.fs.PrSH.emissivity_wall.fix(0.7)  # wall emissivity
    # correction factor for overall heat transfer coefficient
    m.fs.PrSH.fcorrection_htc.fix(1.5)
    # correction factor for pressure drop calc tube side
    m.fs.PrSH.fcorrection_dp_tube.fix(1.0)
    # correction factor for pressure drop calc shell side
    m.fs.PrSH.fcorrection_dp_shell.fix(1.0)

    # Initialize the PrSH unit
    m.fs.PrSH.initialize(outlvl=logging.INFO)
    # scale the PrSH unit
    prsh_scaler=autoscale_unit(m.fs.PrSH,"PrSH")

    # --------- Platen Superheater ------------------------------------------
    # propagate the PrSH outlet state to the PlSH inlet
    _set_port(arc=m.fs.prsh_to_plsh)
    # Fix the PlSH heat duty
    m.fs.PlSH.heat_duty[:].fix(5.5e7)
    # initialize the PlSH unit
    m.fs.PlSH.initialize(outlvl=logging.INFO)
    # scale the PlSH unit
    plsh_scaler=autoscale_unit(m.fs.PlSH,"PlSH")

    #  -------- Finishing Superheater ----------------------------------------
    # propagate the PlSH outlet state to the FSH cold-side inlet
    _set_port(arc=m.fs.plsh_to_fsh)

    # FLUE GAS Inlet from Primary Superheater
    FGrate = 21290.6999  # mol/s
    # Use FG molar composition to set component flow rates (baseline report)
    for j,x in comp.items(): 
        m.fs.FSH.hot_side_inlet.flow_mol_comp[0,j].fix(FGrate*x)
    m.fs.FSH.hot_side_inlet.temperature[0].fix(1300.335)
    m.fs.FSH.hot_side_inlet.pressure[0].fix(100145)

    # Finishing Superheater
    ITM = 0.0254  # inch to meter conversion
    m.fs.FSH.tube_di.fix((2.5 - 2 * 0.165) * ITM)
    m.fs.FSH.tube_thickness.fix(0.165 * ITM)
    m.fs.FSH.pitch_x.fix(3 * ITM)
    # gas path transverse width 54.78 ft / number of columns
    m.fs.FSH.pitch_y.fix(54.78 / 108 * 12 * ITM)
    m.fs.FSH.tube_length.fix(53.13 * 12 * ITM)
    m.fs.FSH.tube_nrow.fix(8 * 2)
    m.fs.FSH.tube_ncol.fix(85)
    m.fs.FSH.nrow_inlet.fix(2)
    m.fs.FSH.delta_elevation.fix(50)
    m.fs.FSH.tube_r_fouling = 0.000176  # (0.001 h-ft^2-F/BTU)
    m.fs.FSH.shell_r_fouling = 0.003131  # (0.03131 - 0.1779 h-ft^2-F/BTU)
    if m.fs.FSH.config.has_radiation is True:
        m.fs.FSH.emissivity_wall.fix(0.7)  # wall emissivity
    # correction factor for overall heat transfer coefficient
    m.fs.FSH.fcorrection_htc.fix(1.0)
    # correction factor for pressure drop calc tube side
    m.fs.FSH.fcorrection_dp_tube.fix(1.0)
    # correction factor for pressure drop calc shell side
    m.fs.FSH.fcorrection_dp_shell.fix(1.0)

    # Initialize the FSH unit
    m.fs.FSH.initialize(outlvl=logging.INFO)
    # scale the FSH unit
    fsh_scaler=autoscale_unit(m.fs.FSH,"FSH")

    # --------- Attemperator inputs ------------------------------------------
    # propagate the FSH steam outlet state to the ATMP1 steam inlet
    _set_port(arc=m.fs.fsh_to_atmp1)
    # Fixed SprayWater from FW pump splitter (splitter is needded)
    hatt2 = value(iapws95.htpx(563.15 * pyunits.K, 2.5449e7 * pyunits.Pa))
    m.fs.ATMP1.SprayWater_state[:].flow_mol.fix(0.001)
    m.fs.ATMP1.SprayWater_state[:].pressure.fix(1.22e8)
    m.fs.ATMP1.SprayWater_state[:].enth_mol.fix(hatt2)
    # Initialize the ATMP1 unit
    m.fs.ATMP1.initialize(outlvl=logging.INFO)
    # scale the ATMP1 unit
    atmp_scaler = autoscale_unit(m.fs.ATMP1, "ATMP1")

    # --------- Splitter ----------------------------------------------------
    # splitter flue gas from Finishing SH to Reheater and Primary SH
    m.fs.Spl1.split_fraction[0, "outlet_1"].fix(0.75)  # 0.85)
    # FLUE GAS Inlet from Primary Superheater
    # Propagate the FSH outlet state to the splitter inlet.
    _set_port(arc=m.fs.fg_fsh_to_split)
    for variable in m.fs.Spl1.inlet.vars.values():
        variable.fix()
    # Initialize the splitter unit
    m.fs.Spl1.initialize(outlvl=logging.INFO)
    # scale the splitter unit
    m.fs.Spl1.scaler = autoscale_unit(m.fs.Spl1, "Spl1")

    # ----------Propagate Splitter outlet -----------------------------------
    # Propagate the splitter results before initializing either gas branch.
    # Propogate the splitter outlet state 1 to the RH hot-side inlet
    _set_port(arc=m.fs.fg_fsh2rh)
    # Propogate the splitter outlet state 2 to the PrSH hot-side inlet
    _set_port(arc=m.fs.fg_fsh2PrSH, overwrite_fixed=True)    

    # ----------- Reheater Superheater --------------------------------------
    #   Steam from HP Turbine outlet 
    m.fs.RH.cold_side_inlet.flow_mol[0].fix(21235.27)  # mol/s
    m.fs.RH.cold_side_inlet.enth_mol[0].fix(53942.7569)  # J/mol
    m.fs.RH.cold_side_inlet.pressure[0].fix(3677172.33638)  # Pascals

    # Hold the propagated splitter outlet while the RH is initialized alone.
    for variable in m.fs.RH.hot_side_inlet.vars.values():
        variable.fix()

    # Reheater Superheater
    ITM = 0.0254  # inch to meter conversion
    m.fs.RH.tube_di.fix((2.5 - 2 * 0.165) * ITM)
    m.fs.RH.tube_thickness.fix(0.11 * ITM)
    m.fs.RH.pitch_x.fix(3 * ITM)
    # gas path transverse width 54.08 ft / number of columns
    m.fs.RH.pitch_y.fix(54.08 / 108 * 12 * ITM)
    m.fs.RH.tube_length.fix(53.82 * 12 * ITM)
    m.fs.RH.tube_nrow.fix(18 * 2)
    m.fs.RH.tube_ncol.fix(82 + 70)
    m.fs.RH.nrow_inlet.fix(2)
    m.fs.RH.delta_elevation.fix(30)
    m.fs.RH.tube_r_fouling = 0.000176  # (0.001 h-ft^2-F/BTU)
    m.fs.RH.shell_r_fouling = 0.00088  # (0.03131 - 0.1779 h-ft^2-F/BTU)
    if m.fs.RH.config.has_radiation is True:
        m.fs.RH.emissivity_wall.fix(0.7)  # wall emissivity
    # correction factor for overall heat transfer coefficient
    m.fs.RH.fcorrection_htc.fix(1.8)
    # correction factor for pressure drop calc tube side
    m.fs.RH.fcorrection_dp_tube.fix(1.0)
    # correction factor for pressure drop calc shell side
    m.fs.RH.fcorrection_dp_shell.fix(1.0)

    # Initialize the RH unit
    m.fs.RH.initialize(outlvl=logging.INFO)
    # scale the RH unit
    rh_scaler = autoscale_unit(m.fs.RH, "RH")

    # Reinitialize PrSH with the propagated splitter branch. Its steam-side inlet
    # remains the initialized Water_wall outlet from the earlier steam-path pass.
    u = m.fs.PrSH
    u.initialize(outlvl=logging.INFO)    

    # Refresh the downstream steam path because the updated PrSH gas inlet can
    # change its cold-side outlet and, consequently, the FSH outlet states.
    _set_port(arc=m.fs.prsh2plsh)
    m.fs.PlSH.initialize(outlvl=logging.INFO)
    _set_port(arc=m.fs.plsh2fsh)
    m.fs.FSH.initialize(outlvl=logging.INFO)

    # Refresh the splitter and both gas branches once after the steam-path update.
    _set_port(arc=m.fs.fg_fsh2_separator, overwrite_fixed=True)
    m.fs.Spl1.initialize(outlvl=logging.INFO)
    _set_port(arc=m.fs.fg_fsh2rh, overwrite_fixed=True)
    m.fs.RH.initialize(outlvl=logging.INFO)
    _set_port(arc=m.fs.fg_fsh2PrSH, overwrite_fixed=True)
    m.fs.PrSH.initialize(outlvl=logging.INFO)


    # Initialize the gas mixer from the refreshed RH and PrSH outlets.
    for inlet, outlet in (
        (m.fs.mix1.Reheat_out, m.fs.RH.hot_side_outlet),
        (m.fs.mix1.PrSH_out, m.fs.PrSH.hot_side_outlet),
    ):
        for j in comp:
            inlet.flow_mol_comp[0, j].fix(value(outlet.flow_mol_comp[0, j]))
        inlet.temperature[0].fix(value(outlet.temperature[0]))
        inlet.pressure[0].fix(value(outlet.pressure[0]))
    # Initialize the mixer unit
    m.fs.mix1.initialize(outlvl=logging.INFO)
    # scale the mixer unit
    mix_scaler = autoscale_unit(m.fs.mix1, "mix1")

    # Release states determined by connected flue-gas arcs. FSH hot-side inlet
    # remains the external boiler-gas boundary; RH cold-side inlet remains the
    # external HP-turbine-exhaust boundary.
    connected_inlet_ports = (
        m.fs.Spl1.inlet,
        m.fs.RH.hot_side_inlet,
        m.fs.PrSH.hot_side_inlet,
        m.fs.mix1.Reheat_out,
        m.fs.mix1.PrSH_out,
        m.fs.ECON.hot_side_inlet,
    )
    for port in connected_inlet_ports:
        for variable in port.vars.values():
            variable.unfix()

    # Populate consistent starting values along the gas path.
    _set_port(arc=m.fs.fg_fsh_to_split)
    _set_port(arc=m.fs.fg_split_to_rh)
    _set_port(arc=m.fs.fg_split_to_prsh)
    _set_port(arc=m.fs.fg_rh_to_mix)
    _set_port(arc=m.fs.fg_prsh_to_mix)
    _set_port(arc=m.fs.fg_mix_to_econ)

    #------------------------------------------------------------------------
    print("initialization done")

def scale_solve(m):
    print("Full flowsheet DOF:",degrees_of_freedom(m.fs))
    # make sure the flowsheet has 0 degrees of freedom before solving
    if degrees_of_freedom(m.fs)!=0: raise RuntimeError("Connected flowsheet is not square")
    # scale the entire flowsheet, but do not overwrite existing factors
    fs_scaler=AutoScaler(overwrite=False) 
    # scale variables by magnitude, but do not overwrite existing factors
    fs_scaler.scale_variables_by_magnitude(m.fs,descend_into=True) 
    # scale constraints by Jacobian norm, but do not overwrite existing factors
    fs_scaler.scale_constraints_by_jacobian_norm(m.fs,norm=2,descend_into=True)
    print("Unscaled active variables:",len(list_unscaled_variables(m.fs,descend_into=True,include_fixed=False)))
    print("Unscaled constraints:",len(list_unscaled_constraints(m.fs,descend_into=True)))

    #solve the connected flowsheet with the updated flue-gas path and refreshed steam path.
    solver=get_solver()
    results=solver.solve(m.fs,tee=True)
    print(results.solver.status,results.solver.termination_condition)
    if not check_optimal_termination(results): raise RuntimeError("Connected solve failed")

    return results

def pfd_result(outfile, m, df):
    tags = {}
    for i in df.index:
        tags[i + "_F"] = df.loc[i, "Molar Flow"]
        tags[i + "_T"] = df.loc[i, "T"]
        tags[i + "_P"] = df.loc[i, "P"]
        tags[i + "_X"] = df.loc[i, "Vapor Fraction"]

    tags["FG_2_RH_Fm"] = value(m.fs.RH.hot_side.properties_in[0].flow_mass)
    tags["FG_2_RH_T"] = value(m.fs.RH.hot_side.properties_in[0].temperature)
    tags["FG_2_RH_P"] = value(m.fs.RH.hot_side.properties_in[0].pressure)

    tags["FG_RH_2_Mix_Fm"] = value(m.fs.RH.hot_side.properties_out[0].flow_mass)
    tags["FG_RH_2_Mix_T"] = value(m.fs.RH.hot_side.properties_out[0].temperature)
    tags["FG_RH_2_Mix_P"] = value(m.fs.RH.hot_side.properties_out[0].pressure)

    tags["FG_2_FSH_Fm"] = value(m.fs.FSH.hot_side.properties_in[0].flow_mass)
    tags["FG_2_FSH_T"] = value(m.fs.FSH.hot_side.properties_in[0].temperature)
    tags["FG_2_FSH_P"] = value(m.fs.FSH.hot_side.properties_in[0].pressure)

    tags["FG_2_PrSH_Fm"] = value(m.fs.PrSH.hot_side.properties_in[0].flow_mass)
    tags["FG_2_PrSH_T"] = value(m.fs.PrSH.hot_side.properties_in[0].temperature)
    tags["FG_2_PrSH_P"] = value(m.fs.PrSH.hot_side.properties_in[0].pressure)

    tags["FG_PrSH_2_Mix_Fm"] = value(m.fs.PrSH.hot_side.properties_out[0].flow_mass)
    tags["FG_PrSH_2_Mix_T"] = value(m.fs.PrSH.hot_side.properties_out[0].temperature)
    tags["FG_PrSH_2_Mix_P"] = value(m.fs.PrSH.hot_side.properties_out[0].pressure)

    tags["FG_2_ECON_Fm"] = value(m.fs.ECON.hot_side.properties_in[0].flow_mass)
    tags["FG_2_ECON_T"] = value(m.fs.ECON.hot_side.properties_in[0].temperature)
    tags["FG_2_ECON_P"] = value(m.fs.ECON.hot_side.properties_in[0].pressure)

    tags["FG_2_AIRPH_Fm"] = value(m.fs.ECON.hot_side.properties_out[0].flow_mass)
    tags["FG_2_AIRPH_T"] = value(m.fs.ECON.hot_side.properties_out[0].temperature)
    tags["FG_2_AIRPH_P"] = value(m.fs.ECON.hot_side.properties_out[0].pressure)

    tags["FG_2_STACK_Fm"] = value(m.fs.ECON.hot_side.properties_out[0].flow_mass)
    tags["FG_2_STACK_T"] = value(m.fs.ECON.hot_side.properties_out[0].temperature)
    tags["FG_2_STACK_P"] = value(m.fs.ECON.hot_side.properties_out[0].pressure)

    original_svg_file = os.path.join(this_file_dir(), "Boiler_scpc_PFD.svg")
    with open(original_svg_file, "r") as f:
        svg_tag(tags, f, outfile=outfile)

def _stream_dict(m):
    """Adds _streams to m, which contains a dictionary of streams for display

    Args:
        m (ConcreteModel): A Pyomo model from create_model()

    Returns:
        None
    """
    # We control m
    # pylint: disable-next=protected-access
    m._streams = OrderedDict(
        [
            ("MS", m.fs.ATMP1.mixed_state),
            ("ATMP_In", m.fs.FSH.cold_side.properties_out),
            ("FSH_In", m.fs.FSH.cold_side.properties_in),
            ("PrSH_IN", m.fs.PrSH.cold_side.properties_in),
            ("RHT_COLD", m.fs.RH.cold_side.properties_in),
            ("RHT_HOT", m.fs.RH.cold_side.properties_out),
            ("PlatenSH_IN", m.fs.PlSH.control_volume.properties_in),
            ("BFW", m.fs.ECON.cold_side.properties_in),
            ("ECON_OUT", m.fs.ECON.cold_side.properties_out),
        ]
    )

def print_results(m):
    print()
    print("Results")
    print()

    print("viscosity gas side = ", m.fs.PrSH.hot_side.properties_in[0].visc_d.value)
    print(
        "conductivity gas side = ", m.fs.PrSH.hot_side.properties_in[0].therm_cond.value
    )
    print("velocity_tube = ", m.fs.PrSH.v_tube[0].value)
    print("velocity_shell = ", m.fs.PrSH.v_shell[0].value)
    print("Re_tube = ", m.fs.PrSH.N_Re_tube[0].value)
    print("Re_shell = ", m.fs.PrSH.N_Re_shell[0].value)
    print("hconv_tube = ", m.fs.PrSH.hconv_tube[0].value)
    print("hconv_shell_rad = ", m.fs.PrSH.hconv_shell_rad[0].value)
    print("hconv_shell_conv = ", m.fs.PrSH.hconv_shell_conv[0].value)
    print("hconv_shell_total = ", m.fs.PrSH.hconv_shell_total[0].value)
    print("driving force = ", value(m.fs.PrSH.delta_temperature[0]))
    print("dT_inlet = ", m.fs.PrSH.deltaT_1[0].value)
    print("dT_outlet = ", m.fs.PrSH.deltaT_2[0].value)
    print("deltaP tube = ", m.fs.PrSH.deltaP_tube[0].value)
    print("deltaP shell = ", m.fs.PrSH.deltaP_shell[0].value)
    print("mbl = ", value(m.fs.PrSH.mbl))

    if m.fs.PrSH.config.has_radiation is True:
        print("gas emissivity = ", m.fs.PrSH.gas_emissivity[0].value)
        print("gas emissivity div2 = ", m.fs.PrSH.gas_emissivity_div2[0].value)
        print("gas emissivity mul2 = ", m.fs.PrSH.gas_emissivity_mul2[0].value)
        print("gas gray fraction = ", m.fs.PrSH.gas_gray_fraction[0].value)
    print(
        "liquid density in = ",
        value(m.fs.PrSH.cold_side.properties_in[0].dens_mass_phase["Liq"]),
    )

    print(
        "liquid density out = ",
        value(m.fs.PrSH.cold_side.properties_out[0].dens_mass_phase["Liq"]),
    )
    print("heat transfer area = ", value(m.fs.PrSH.area))
    print(
        "overall heat transfer = ",
        value(m.fs.PrSH.overall_heat_transfer_coefficient[0]),
    )

    print("\n\n ------------- Economizer   ---------")
    print("liquid temp in = ", value(m.fs.ECON.cold_side.properties_in[0].temperature))
    print("liquid temp out = ", value(m.fs.ECON.cold_side.properties_out[0].temperature))
    print("gas temp in = ", value(m.fs.ECON.hot_side.properties_in[0].temperature))
    print("gas temp out = ", value(m.fs.ECON.hot_side.properties_out[0].temperature))

    print("\n\n ------------- water wall  ---------")
    print(
        "liquid temp in = ",
        value(m.fs.Water_wall.control_volume.properties_in[0].temperature),
    )
    print(
        "steam temp out = ",
        value(m.fs.Water_wall.control_volume.properties_out[0].temperature),
    )

    print("\n\n ------------- Primary Superheater  ---------")
    print("steam temp in = ", value(m.fs.PrSH.cold_side.properties_in[0].temperature))
    print("steam temp out = ", value(m.fs.PrSH.cold_side.properties_out[0].temperature))
    print("gas temp in = ", value(m.fs.PrSH.hot_side.properties_in[0].temperature))
    print("gas temp out = ", value(m.fs.PrSH.hot_side.properties_out[0].temperature))

    print("\n\n ------------- Platen SH  ---------")
    print(
        "steam temp in = ", value(m.fs.PlSH.control_volume.properties_in[0].temperature)
    )
    print(
        "steam temp out = ",
        value(m.fs.PlSH.control_volume.properties_out[0].temperature),
    )

    print("\n\n ------------- Finishing Superheater  ---------")
    print("steam temp in = ", value(m.fs.FSH.cold_side.properties_in[0].temperature))
    print(
        "steam temp out (to attmp) = ",
        value(m.fs.FSH.cold_side.properties_out[0].temperature),
    )
    print("gas temp in = ", value(m.fs.FSH.hot_side.properties_in[0].temperature))
    print("gas temp out = ", value(m.fs.FSH.hot_side.properties_out[0].temperature))

    print("\n\n ------------- Attemperator  ---------")
    print(
        "steam temp in = ",
        value(m.fs.ATMP1.Steam.enth_mol[0]),
        value(m.fs.FSH.cold_side.properties_out[0].temperature),
    )
    print(
        "steam temp out (to HP turbine) = ",
        value(m.fs.ATMP1.outlet.enth_mol[0]),
        value(m.fs.ATMP1.mixed_state[0].temperature),
    )
    print()

    print("\n\n ------------- Reheater  ---------")
    print("liquid temp in = ", value(m.fs.RH.cold_side.properties_in[0].temperature))
    print("liquid temp out = ", value(m.fs.RH.cold_side.properties_out[0].temperature))
    print("gas temp in = ", value(m.fs.RH.hot_side.properties_in[0].temperature))
    print("gas temp out = ", value(m.fs.RH.hot_side.properties_out[0].temperature))

if __name__ == "__main__":
    # generate the model and solver
    m, solver = main()
    # set solver options
    solver.options = {
        "tol": 1e-6,
        "linear_solver": "ma27",
        "max_iter": 100,
        "halt_on_ampl_error": "yes",
    }
    # initialize the model
    initialize(m)
    # Scale and solve the model
    results = scale_solve(m)
    # print the results
    print_results(m)
