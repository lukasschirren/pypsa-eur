# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Description
-----------

.. tip::
    The rule :mod:`prepare_elec_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    get,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import load_costs, set_transmission_costs

# Allow for PyPSA versions <0.35
if PYPSA_V1:
    from pypsa.common import expand_series
else:
    from pypsa.descriptors import expand_series


idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def modify_attribute(n, adjustments, investment_year, modification="factor"):
    if not adjustments[modification]:
        return
    change_dict = adjustments[modification]
    for c in change_dict.keys():
        if c not in n.component_attrs.keys():
            logger.warning(f"{c} needs to be a PyPSA Component")
            continue
        for carrier in change_dict[c].keys():
            ind_i = n.df(c)[n.df(c).carrier == carrier].index
            if ind_i.empty:
                continue
            for parameter in change_dict[c][carrier].keys():
                if parameter not in n.df(c).columns:
                    logger.warning(f"Attribute {parameter} needs to be in {c} columns.")
                    continue
                if investment_year:
                    factor = get(change_dict[c][carrier][parameter], investment_year)
                else:
                    factor = change_dict[c][carrier][parameter]
                if modification == "factor":
                    logger.info(f"Modify {parameter} of {carrier} by factor {factor} ")
                    n.df(c).loc[ind_i, parameter] *= factor
                elif modification == "absolute":
                    logger.info(f"Set {parameter} of {carrier} to {factor} ")
                    n.df(c).loc[ind_i, parameter] = factor
                else:
                    logger.warning(
                        f"{modification} needs to be either 'absolute' or 'factor'."
                    )


def maybe_adjust_costs_and_potentials(n, adjustments, investment_year=None):
    if not adjustments:
        return
    for modification in adjustments.keys():
        modify_attribute(n, adjustments, investment_year, modification)


def add_co2limit(n, co2limit, Nyears=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2limit * Nyears,
    )


def add_gaslimit(n, gaslimit, Nyears=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * Nyears,
    )


def add_emission_prices(n, emission_prices={"co2": 0.0}, regional_prices=None, exclude_co2=False):
    """
    Add emission prices to generators and storage units.

    Parameters
    ----------
    n : pypsa.Network
        Network to add emission prices to
    emission_prices : dict, optional
        Uniform emission prices for all regions, by default {"co2": 0.0}
    regional_prices : dict, optional
        Regional emission prices, by default None
        Structure: {"country_code": {"co2": price}, ...}
        Example: {"DE": {"co2": 60.0}, "FR": {"co2": 45.0}}
    exclude_co2 : bool, optional
        Whether to exclude CO2 prices, by default False
    """
    if exclude_co2:
        if regional_prices:
            for region in regional_prices:
                regional_prices[region].pop("co2", None)
        else:
            emission_prices.pop("co2", None)

    if regional_prices:
        logger.info("Applying regional emission prices")
        
        # Apply regional prices
        for country, prices in regional_prices.items():
            # Find generators in this country
            country_buses = n.buses.index[n.buses.country == country]
            gen_mask = n.generators.bus.isin(country_buses)
            su_mask = n.storage_units.bus.isin(country_buses) if not n.storage_units.empty else pd.Series(dtype=bool)
            
            if gen_mask.any():
                # Calculate emission prices for this region
                ep = (
                    pd.Series(prices).rename(lambda x: x + "_emissions")
                    * n.carriers.filter(like="_emissions")
                ).sum(axis=1)
                
                # Apply to generators - same logic as uniform pricing
                gen_ep = n.generators.loc[gen_mask, "carrier"].map(ep) / n.generators.loc[gen_mask, "efficiency"]
                n.generators.loc[gen_mask, "marginal_cost"] += gen_ep
                
                # Apply to time-varying marginal costs - same logic as uniform pricing
                if not n.generators_t["marginal_cost"].empty:
                    gen_cols = n.generators_t["marginal_cost"].columns.intersection(gen_ep.index)
                    if len(gen_cols) > 0:
                        n.generators_t["marginal_cost"].loc[:, gen_cols] += gen_ep[gen_cols]
                
                logger.info(f"Applied emission prices to {gen_mask.sum()} generators in {country}: {prices}")
            
            # Apply to storage units
            if not n.storage_units.empty and su_mask.any():
                ep = (
                    pd.Series(prices).rename(lambda x: x + "_emissions")
                    * n.carriers.filter(like="_emissions")
                ).sum(axis=1)
                su_ep = n.storage_units.loc[su_mask, "carrier"].map(ep) / n.storage_units.loc[su_mask, "efficiency_dispatch"]
                n.storage_units.loc[su_mask, "marginal_cost"] += su_ep
    else:
        # Apply uniform prices (original logic)
        ep = (
            pd.Series(emission_prices).rename(lambda x: x + "_emissions")
            * n.carriers.filter(like="_emissions")
        ).sum(axis=1)
        gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
        n.generators["marginal_cost"] += gen_ep
        n.generators_t["marginal_cost"] += gen_ep[n.generators_t["marginal_cost"].columns]
        if not n.storage_units.empty:
            su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
            n.storage_units["marginal_cost"] += su_ep


def add_dynamic_emission_prices(n, fn, regional_prices_files=None):
    """
    Add time-varying emission prices to generators.

    Parameters
    ----------
    n : pypsa.Network
        Network to add emission prices to
    fn : str
        Path to CSV file with time-varying CO2 prices (for uniform pricing)
    regional_prices_files : dict, optional
        Dictionary mapping country codes to CSV files with regional time-varying prices
        Structure: {"country_code": "path/to/co2_prices.csv", ...}
        Example: {"DE": "data/co2_prices_de.csv", "FR": "data/co2_prices_fr.csv"}
    """
    if regional_prices_files:
        logger.info("Applying regional time-varying emission prices")
        
        static = n.generators.marginal_cost
        dynamic = n.get_switchable_as_dense("Generator", "marginal_cost")
        
        for country, price_file in regional_prices_files.items():
            # Load regional price data
            co2_price = pd.read_csv(price_file, index_col=0, parse_dates=True)
            co2_price = co2_price[~co2_price.index.duplicated()]
            co2_price = co2_price.reindex(n.snapshots).ffill().bfill()
            
            # Find generators in this country
            country_buses = n.buses.index[n.buses.country == country]
            gen_mask = n.generators.bus.isin(country_buses)
            
            if gen_mask.any():
                # Calculate emissions for generators in this region - same as uniform logic
                emissions = (
                    n.generators.loc[gen_mask, "carrier"].map(n.carriers.co2_emissions) 
                    / n.generators.loc[gen_mask, "efficiency"]
                )
                
                # Create time-varying costs for this region - same as uniform logic
                co2_cost = expand_series(emissions, n.snapshots).T.mul(co2_price.iloc[:, 0], axis=0)
                
                # Add to dynamic marginal costs - co2_cost.columns are the generator names
                gen_cols = dynamic.columns.intersection(co2_cost.columns)
                if len(gen_cols) > 0:
                    dynamic.loc[:, gen_cols] += co2_cost[gen_cols]
                
                logger.info(f"Applied time-varying emission prices to {gen_mask.sum()} generators in {country}")
        
        # Update network with combined dynamic costs
        marginal_cost = dynamic
        n.generators_t.marginal_cost = marginal_cost.loc[:, marginal_cost.ne(static).any()]
        
    else:
        # Apply uniform time-varying prices (original logic)
        co2_price = pd.read_csv(fn, index_col=0, parse_dates=True)
        co2_price = co2_price[~co2_price.index.duplicated()]
        co2_price = co2_price.reindex(n.snapshots).ffill().bfill()

        emissions = (
            n.generators.carrier.map(n.carriers.co2_emissions) / n.generators.efficiency
        )
        co2_cost = expand_series(emissions, n.snapshots).T.mul(co2_price.iloc[:, 0], axis=0)

        static = n.generators.marginal_cost
        dynamic = n.get_switchable_as_dense("Generator", "marginal_cost")

        marginal_cost = dynamic + co2_cost.reindex(columns=dynamic.columns, fill_value=0)
        n.generators_t.marginal_cost = marginal_cost.loc[:, marginal_cost.ne(static).any()]


def set_line_s_max_pu(n, s_max_pu=0.7):
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


def set_transmission_limit(n, kind, factor, costs, Nyears=1):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if kind == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    set_transmission_costs(n, costs)

    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    if factor != "opt":
        con_type = "expansion_cost" if kind == "c" else "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            f"l{kind}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    return n


def average_every_nhours(n, offset, drop_leap_day=False):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(snapshots=[])

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    sns = snapshot_weightings.index
    if drop_leap_day:
        sns = sns[~((sns.month == 2) & (sns.day == 29))]
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


def apply_time_segmentation(n, segments, solver_name="cbc"):
    logger.info(f"Aggregating time series to {segments} segments.")
    try:
        import tsam.timeseriesaggregation as tsam
    except ImportError:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found.Install via 'pip install tsam'"
        )

    p_max_pu_norm = n.generators_t.p_max_pu.max()
    p_max_pu = n.generators_t.p_max_pu / p_max_pu_norm

    load_norm = n.loads_t.p_set.max()
    load = n.loads_t.p_set / load_norm

    inflow_norm = n.storage_units_t.inflow.max()
    inflow = n.storage_units_t.inflow / inflow_norm

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=solver_name,
    )

    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

    n.set_snapshots(pd.DatetimeIndex(snapshots, name="name"))
    n.snapshot_weightings = pd.Series(
        weightings, index=snapshots, name="weightings", dtype="float64"
    )

    segmented.index = snapshots
    n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
    n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
    n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

    return n


def enforce_autarky(n, only_crossborder=False):
    if only_crossborder:
        lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index
    n.remove("Line", lines_rm)
    n.remove("Link", links_rm)


def override_interconnection_capacities(n, override_config, investment_year=None):
    """
    Override cross-border interconnection capacities with absolute MW values.
    Mirrors the same function in prepare_sector_network.py so the electricity
    pipeline can also apply UA interconnection overrides.
    """
    if not override_config:
        return

    if isinstance(override_config, list):
        overrides = override_config
        apply_years = []
    else:
        apply_years = override_config.get("apply_years", [])
        overrides = override_config.get("overrides", [])

    if not overrides:
        return

    if apply_years and investment_year is not None:
        if investment_year not in apply_years:
            logger.info(
                f"Skipping interconnection capacity overrides: "
                f"year {investment_year} not in apply_years {apply_years}"
            )
            return

    logger.info(
        f"Applying {len(overrides)} interconnection capacity overrides "
        f"(year={investment_year})"
    )

    for entry in overrides:
        b0 = entry.get("bus0")
        b1 = entry.get("bus1")
        s_nom = entry.get("s_nom")
        p_nom = entry.get("p_nom")

        if b0 is None or b1 is None:
            logger.warning(f"  Skipping override with missing bus0/bus1: {entry}")
            continue

        if s_nom is not None and not n.lines.empty:
            mask = (
                ((n.lines.bus0 == b0) & (n.lines.bus1 == b1))
                | ((n.lines.bus0 == b1) & (n.lines.bus1 == b0))
            )
            if mask.any():
                old_val = n.lines.loc[mask, "s_nom"].values
                n.lines.loc[mask, "s_nom"] = s_nom
                n.lines.loc[mask, "s_nom_min"] = s_nom
                if "s_nom_max" in n.lines.columns:
                    n.lines.loc[mask, "s_nom_max"] = n.lines.loc[
                        mask, "s_nom_max"
                    ].clip(lower=s_nom)
                logger.info(
                    f"  AC line {b0} <-> {b1}: {old_val} -> {s_nom} MW ({mask.sum()} line(s))"
                )
            else:
                logger.warning(f"  No AC line found for {b0} <-> {b1}")

        if p_nom is not None and not n.links.empty:
            mask = (
                ((n.links.bus0 == b0) & (n.links.bus1 == b1))
                | ((n.links.bus0 == b1) & (n.links.bus1 == b0))
            )
            if mask.any():
                old_val = n.links.loc[mask, "p_nom"].values
                n.links.loc[mask, "p_nom"] = p_nom
                n.links.loc[mask, "p_nom_min"] = p_nom
                if "p_nom_max" in n.links.columns:
                    n.links.loc[mask, "p_nom_max"] = n.links.loc[
                        mask, "p_nom_max"
                    ].clip(lower=p_nom)
                logger.info(
                    f"  DC link {b0} <-> {b1}: {old_val} -> {p_nom} MW ({mask.sum()} link(s))"
                )
            else:
                logger.warning(f"  No DC link found for {b0} <-> {b1}")


def set_line_nom_max(
    n,
    s_nom_max_set=np.inf,
    p_nom_max_set=np.inf,
    s_nom_max_ext=np.inf,
    p_nom_max_ext=np.inf,
):
    if np.isfinite(s_nom_max_ext) and s_nom_max_ext > 0:
        logger.info(f"Limiting line extensions to {s_nom_max_ext} MW")
        n.lines["s_nom_max"] = n.lines["s_nom"] + s_nom_max_ext

    if np.isfinite(p_nom_max_ext) and p_nom_max_ext > 0:
        logger.info(f"Limiting link extensions to {p_nom_max_ext} MW")
        hvdc = n.links.index[n.links.carrier == "DC"]
        n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"] + p_nom_max_ext

    n.lines["s_nom_max"] = n.lines.s_nom_max.clip(upper=s_nom_max_set)
    n.links["p_nom_max"] = n.links.p_nom_max.clip(upper=p_nom_max_set)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network",
            clusters="37",
            opts="Co2L-4H",
        )
    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.params.costs,
        snakemake.params.max_hours,
        Nyears,
    )

    set_line_s_max_pu(n, snakemake.params.lines["s_max_pu"])

    # temporal averaging
    time_resolution = snakemake.params.time_resolution
    is_string = isinstance(time_resolution, str)
    if is_string and time_resolution.lower().endswith("h"):
        n = average_every_nhours(n, time_resolution, snakemake.params.drop_leap_day)

    # segments with package tsam
    if is_string and time_resolution.lower().endswith("seg"):
        solver_name = snakemake.config["solving"]["solver"]["name"]
        segments = int(time_resolution.replace("seg", ""))
        n = apply_time_segmentation(n, segments, solver_name)

    if snakemake.params.co2limit_enable:
        add_co2limit(n, snakemake.params.co2limit, Nyears)

    if snakemake.params.gaslimit_enable:
        add_gaslimit(n, snakemake.params.gaslimit, Nyears)

    maybe_adjust_costs_and_potentials(n, snakemake.params["adjustments"])

    emission_prices = snakemake.params.costs["emission_prices"]
    
    # Check if regional pricing is enabled
    regional_config = emission_prices.get("regional_prices", {})
    if regional_config.get("enable", False):
        logger.info("Regional emission pricing enabled")
        
        # Handle regional time-varying prices
        if regional_config.get("monthly_prices_files") and any(regional_config["monthly_prices_files"].values()):
            logger.info("Setting regional time-varying emission prices")
            add_dynamic_emission_prices(n, None, regional_prices_files=regional_config["monthly_prices_files"])
        
        # Handle regional static prices
        elif regional_config.get("prices"):
            logger.info("Setting regional static emission prices")
            add_emission_prices(n, regional_prices=regional_config["prices"])
        else:
            logger.warning("Regional emission pricing enabled but no prices configured")
    
    # Standard uniform pricing (backward compatibility)
    elif emission_prices["co2_monthly_prices"]:
        logger.info(
            "Setting time dependent emission prices according spot market price"
        )
        add_dynamic_emission_prices(n, snakemake.input.co2_price)
    elif emission_prices["enable"]:
        add_emission_prices(
            n, dict(co2=snakemake.params.costs["emission_prices"]["co2"])
        )

    kind = snakemake.params.transmission_limit[0]
    factor = snakemake.params.transmission_limit[1:]
    set_transmission_limit(n, kind, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.params.lines.get("s_nom_max", np.inf),
        p_nom_max_set=snakemake.params.links.get("p_nom_max", np.inf),
        s_nom_max_ext=snakemake.params.lines.get("max_extension", np.inf),
        p_nom_max_ext=snakemake.params.links.get("max_extension", np.inf),
    )

    if snakemake.params.autarky["enable"]:
        only_crossborder = snakemake.params.autarky["by_country"]
        enforce_autarky(n, only_crossborder=only_crossborder)

    override_interconnection_capacities(
        n,
        snakemake.params.get("interconnection_capacity_override", {}),
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
