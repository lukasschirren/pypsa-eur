# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    sanitize_custom_columns,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import flatten, sanitize_carriers
from scripts.add_existing_baseyear import add_build_year_to_new_assets

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def add_brownfield(
    n,
    n_p,
    year,
    h2_retrofit=False,
    h2_retrofit_capacity_per_ch4=None,
    capacity_threshold=None,
):
    """
    Add brownfield capacity from previous network.

    Parameters
    ----------
    n : pypsa.Network
        Network to add brownfield to
    n_p : pypsa.Network
        Previous network to get brownfield from
    year : int
        Planning year
    h2_retrofit : bool
        Whether to allow hydrogen pipeline retrofitting
    h2_retrofit_capacity_per_ch4 : float
        Ratio of hydrogen to methane capacity for pipeline retrofitting
    capacity_threshold : float
        Threshold for removing assets with low capacity
    """
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.remove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime <= year
        n_p.remove(c.name, c.df.index[c.df.build_year + c.df.lifetime <= year])

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[
            (c.df[f"{attr}_nom_extendable"] & c.df.index.str.contains("urban central"))
            & c.df.index.str.contains("CHP")
            & c.df.index.str.contains("heat")
        ]

        if not chp_heat.empty:
            threshold_chp_heat = (
                capacity_threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.remove(
                c.name,
                chp_heat[c.df.loc[chp_heat, f"{attr}_nom_opt"] < threshold_chp_heat],
            )

        n_p.remove(
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & ~c.df.index.isin(chp_heat))
                & (c.df[f"{attr}_nom_opt"] < capacity_threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.add(c.name, c.df.index, **c.df)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            # TODO: Needs to be rewritten to
            n._import_series_from_df(c.pnl[tattr], c.name, tattr)

    # deal with gas network
    if h2_retrofit:
        # subtract the already retrofitted from the maximum capacity
        h2_retrofitted_fixed_i = n.links[
            (n.links.carrier == "H2 pipeline retrofitted")
            & (n.links.build_year != year)
        ].index
        h2_retrofitted = n.links[
            (n.links.carrier == "H2 pipeline retrofitted")
            & (n.links.build_year == year)
        ].index

        # pipe capacity always set in prepare_sector_network to todays gas grid capacity * H2_per_CH4
        # and is therefore constant up to this point
        pipe_capacity = n.links.loc[h2_retrofitted, "p_nom_max"]
        # already retrofitted capacity from gas -> H2
        already_retrofitted = (
            n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
            .rename(lambda x: x.split("-2")[0] + f"-{year}")
            .groupby(level=0)
            .sum()
        )
        remaining_capacity = pipe_capacity - already_retrofitted.reindex(
            index=pipe_capacity.index
        ).fillna(0)
        n.links.loc[h2_retrofitted, "p_nom_max"] = remaining_capacity

        # reduce gas network capacity
        gas_pipes_i = n.links[n.links.carrier == "gas pipeline"].index
        if not gas_pipes_i.empty:
            # subtract the already retrofitted from today's gas grid capacity
            pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
            fr = "H2 pipeline retrofitted"
            to = "gas pipeline"
            CH4_per_H2 = 1 / h2_retrofit_capacity_per_ch4
            already_retrofitted.index = already_retrofitted.index.str.replace(fr, to)
            remaining_capacity = (
                pipe_capacity
                - CH4_per_H2
                * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
            )
            n.links.loc[gas_pipes_i, "p_nom"] = remaining_capacity
            n.links.loc[gas_pipes_i, "p_nom_max"] = remaining_capacity

    # --- BF-BOF → gas DRI retrofit: set p_nom_max from existing BF-BOF capacity ---
    # Same brownfield issue as CC retrofit: fresh BF-BOF p_nom=0 in 2030+, so the
    # p_nom_max computed in prepare_sector_network is also 0.  Recompute here from
    # the accumulated brownfield BF-BOF capacity.
    # Conversion (MW_coal → MW_gas):
    #   p_nom_max_gas = p_nom_bfbof × eff_bfbof × EAF_hbi_input / eff_dri_ret
    #   where EAF_hbi_input = -eff2_eaf / eff_eaf
    dri_ret_i = n.links[
        (n.links.carrier == "BF-BOF to DRI retrofit")
        & (n.links.build_year == year)
    ].index

    if not dri_ret_i.empty:
        existing_bfbof_dri_i = n.links[
            (n.links.carrier == "BF-BOF")
            & (n.links.build_year != year)
        ].index

        if not existing_bfbof_dri_i.empty:
            eff_bfbof = n.links.loc[existing_bfbof_dri_i, "efficiency"].iloc[0]
            eff_dri_ret = n.links.loc[dri_ret_i, "efficiency"].iloc[0]

            eaf_i = n.links[n.links.carrier == "EAF"].index
            eff_eaf = n.links.loc[eaf_i, "efficiency"].iloc[0]
            eff2_eaf = n.links.loc[eaf_i, "efficiency2"].iloc[0]
            eaf_hbi_input = -eff2_eaf / eff_eaf  # tHBI consumed per t_steel

            conversion = eff_bfbof * eaf_hbi_input / eff_dri_ret

            # Group existing BF-BOF by node (strip " steel" from bus1)
            pnom_by_node = (
                n.links.loc[existing_bfbof_dri_i]
                .assign(node=lambda df: df["bus1"].str.replace(r" steel$", "", regex=True))
                .groupby("node")["p_nom"]
                .sum()
            )
            # DRI retrofit bus1 = HBI node ("X HBI") → strip " HBI" to get node
            dri_node = n.links.loc[dri_ret_i, "bus1"].str.replace(r" HBI$", "", regex=True)
            new_pnom_max = dri_node.map(pnom_by_node).fillna(0) * conversion
            n.links.loc[dri_ret_i, "p_nom_max"] = new_pnom_max.values

            logger.info(
                f"BF-BOF→DRI retrofit: set p_nom_max for {len(dri_ret_i)} links "
                f"(total {new_pnom_max.sum():.1f} MW_gas from "
                f"{pnom_by_node.sum():.1f} MW_coal existing BF-BOF, "
                f"conversion={conversion:.4f})"
            )
        else:
            logger.info(
                "BF-BOF→DRI retrofit: no existing BF-BOF from previous periods; "
                "p_nom_max remains 0."
            )

    # --- BF-BOF CC retrofit: set p_nom_max from existing BF-BOF capacity ---
    # New-period CC retrofit links are born with p_nom_max=0 because the fresh
    # BF-BOF link in prepare_sector_network has p_nom=0 (brownfield fix).
    # Here we raise p_nom_max to match the CO2 output of accumulated BF-BOF.
    # Conversion: p_nom_max_CC (MW_el) = p_nom_bfbof (MW_coal) × coal_CO2 × el_input_CC
    cc_retrofit_i = n.links[
        (n.links.carrier == "BF-BOF CC retrofit")
        & (n.links.build_year == year)
    ].index

    if not cc_retrofit_i.empty:
        existing_bfbof_i = n.links[
            (n.links.carrier == "BF-BOF")
            & (n.links.build_year != year)
        ].index

        if not existing_bfbof_i.empty:
            # CC efficiency = CC_capture_rate / CC_el_input; BF-BOF efficiency2 = coal_CO2
            coal_co2 = n.links.loc[existing_bfbof_i, "efficiency2"].iloc[0]   # tCO2/MWh_coal
            cc_eff2 = n.links.loc[cc_retrofit_i, "efficiency2"].iloc[0]       # tCO2_removed/MWh_el (negative)
            # p_nom_max_CC = p_nom_bfbof × coal_co2 / (-cc_eff2)
            # = p_nom_bfbof × coal_co2 × (CC_el_input / CC_capture_rate)
            conversion = coal_co2 / (-cc_eff2)

            # Sum existing BF-BOF p_nom grouped by node (strip " steel" from bus1).
            # BF-BOF bus0 is the single global "EU coal" bus, so we must use bus1
            # (per-node steel bus e.g. "DE0 0 steel") to get per-node capacity.
            pnom_by_node = (
                n.links.loc[existing_bfbof_i]
                .assign(node=lambda df: df["bus1"].str.replace(r" steel$", "", regex=True))
                .groupby("node")["p_nom"]
                .sum()
            )

            # CC retrofit bus0 is the per-node electricity bus (== node name).
            cc_bus0 = n.links.loc[cc_retrofit_i, "bus0"]  # electricity bus = node
            new_pnom_max = cc_bus0.map(pnom_by_node).fillna(0) * conversion

            # Subtract already-installed CC from prior vintages so cumulative CC
            # cannot exceed BF-BOF CO2 capacity (mirrors gas DRI→H2 DRI pattern).
            already_cc_i = n.links[
                (n.links.carrier == "BF-BOF CC retrofit")
                & (n.links.build_year != year)
            ].index
            if not already_cc_i.empty:
                already_by_node = (
                    n.links.loc[already_cc_i]
                    .groupby("bus0")["p_nom"]  # bus0 = per-node electricity bus
                    .sum()
                )
                new_pnom_max = (
                    new_pnom_max - cc_bus0.map(already_by_node).fillna(0)
                ).clip(lower=0)

            n.links.loc[cc_retrofit_i, "p_nom_max"] = new_pnom_max.values

            logger.info(
                f"BF-BOF CC retrofit: set p_nom_max for {len(cc_retrofit_i)} links "
                f"(total {new_pnom_max.sum():.1f} MW_el from "
                f"{pnom_by_node.sum():.1f} MW_coal existing BF-BOF, "
                f"conversion={conversion:.4f})"
            )
        else:
            logger.info(
                "BF-BOF CC retrofit: no existing BF-BOF from previous periods; "
                "p_nom_max remains 0."
            )

    # --- Gas DRI → H2 DRI retrofit: set p_nom_max from existing gas DRI capacity ---
    # New retrofit links (created in this period) have p_nom_max=0 from
    # prepare_sector_network.  We raise it to match the gas DRI capacity
    # carried forward from previous periods, after converting MW_gas → MW_H2
    # so that the retrofit can produce the same HBI output.
    retrofit_h2_i = n.links[
        (n.links.carrier == "gas DRI to H2 DRI retrofit")
        & (n.links.build_year == year)
    ].index

    if not retrofit_h2_i.empty:
        # Existing gas DRI from previous periods (fixed by brownfield)
        existing_gas_dri_i = n.links[
            (n.links.carrier == "gas DRI")
            & (n.links.build_year != year)
        ].index

        if not existing_gas_dri_i.empty:
            # Unit conversion: same HBI throughput requires different MW on
            # each fuel bus.  efficiency = 1/fuel_input, so
            # p_nom_H2 = p_nom_gas × eff_gas / eff_H2 = p_nom_gas × fuel_H2/fuel_gas
            gas_dri_eff = n.links.loc[existing_gas_dri_i, "efficiency"].iloc[0]
            h2_retrofit_eff = n.links.loc[retrofit_h2_i, "efficiency"].iloc[0]
            conversion = gas_dri_eff / h2_retrofit_eff  # fuel_input["H2"] / fuel_input["gas"]

            # Sum existing gas DRI p_nom grouped by bus1 (HBI bus = node identity)
            pnom_by_hbi = (
                n.links.loc[existing_gas_dri_i]
                .groupby("bus1")["p_nom"]
                .sum()
            )

            # Subtract already-installed retrofit capacity from earlier periods so
            # cumulative retrofits cannot exceed the original gas DRI plant capacity.
            already_retrofitted_i = n.links[
                (n.links.carrier == "gas DRI to H2 DRI retrofit")
                & (n.links.build_year != year)
            ].index
            if not already_retrofitted_i.empty:
                already_by_hbi = (
                    n.links.loc[already_retrofitted_i]
                    .groupby("bus1")["p_nom"]
                    .sum()
                ) / conversion  # convert MW_H2 → MW_gas to subtract in gas-DRI units
            else:
                already_by_hbi = pd.Series(dtype=float)

            # Map each retrofit link to its HBI bus and set p_nom_max
            retrofit_hbi = n.links.loc[retrofit_h2_i, "bus1"]
            available_gas_cap = (
                pnom_by_hbi - already_by_hbi.reindex(pnom_by_hbi.index).fillna(0)
            ).clip(lower=0)
            new_pnom_max = retrofit_hbi.map(available_gas_cap).fillna(0) * conversion
            n.links.loc[retrofit_h2_i, "p_nom_max"] = new_pnom_max.values

            logger.info(
                f"Gas DRI→H2 DRI retrofit: set p_nom_max for {len(retrofit_h2_i)} links "
                f"(total {new_pnom_max.sum():.1f} MW_H2 from "
                f"{pnom_by_hbi.sum():.1f} MW_gas existing gas DRI minus "
                f"{already_by_hbi.sum():.1f} MW_gas already retrofitted, "
                f"conversion factor={conversion:.4f})"
            )
        else:
            logger.info(
                "Gas DRI→H2 DRI retrofit: no existing gas DRI from previous periods; "
                "p_nom_max remains 0."
            )

    # --- Gas DRI H2 blend: set p_nom_max from existing gas DRI capacity ---
    # The 30% H2 blend option can only run where gas DRI plants already exist.
    # Conversion: running the same tHBI throughput in blend mode consumes only
    # (1 - h2_blend_frac) × fuel_input["gas"] MW_gas, so:
    #   p_nom_max_blend = p_nom_gas × (eff_gas_dri / eff_blend) = p_nom_gas × (1 - h2_blend_frac)
    blend_i = n.links[
        (n.links.carrier == "gas DRI H2 blend")
        & (n.links.build_year == year)
    ].index

    if not blend_i.empty:
        existing_gas_dri_i = n.links[
            (n.links.carrier == "gas DRI")
            & (n.links.build_year != year)
        ].index

        if not existing_gas_dri_i.empty:
            gas_dri_eff = n.links.loc[existing_gas_dri_i, "efficiency"].iloc[0]
            blend_eff = n.links.loc[blend_i, "efficiency"].iloc[0]
            # conversion = eff_gas_dri / eff_blend = gas_per_tHBI / fuel_input["gas"] = (1 - h2_blend_frac)
            conversion = gas_dri_eff / blend_eff

            pnom_by_hbi = (
                n.links.loc[existing_gas_dri_i]
                .groupby("bus1")["p_nom"]
                .sum()
            )

            # Deduct gas DRI capacity already permanently converted to pure H2
            # via the gas DRI → H2 DRI retrofit.  Once a plant runs on 100% H2
            # it can no longer operate in gas-H2 blend mode.
            # Conversion: p_nom_H2_retrofit (MW_H2) → MW_gas equivalent
            #   MW_gas = MW_H2 / (gas_dri_eff / h2ret_eff)
            already_h2ret_i = n.links[
                (n.links.carrier == "gas DRI to H2 DRI retrofit")
                & (n.links.build_year != year)
            ].index
            if not already_h2ret_i.empty:
                h2ret_eff = n.links.loc[already_h2ret_i, "efficiency"].iloc[0]
                gas_to_h2_conv = gas_dri_eff / h2ret_eff  # fuel_H2/fuel_gas (MW_H2 per MW_gas)
                already_h2ret_by_hbi = (
                    n.links.loc[already_h2ret_i].groupby("bus1")["p_nom"].sum()
                ) / gas_to_h2_conv  # MW_H2 → MW_gas equivalent
                pnom_by_hbi = (
                    pnom_by_hbi
                    - already_h2ret_by_hbi.reindex(pnom_by_hbi.index).fillna(0)
                ).clip(lower=0)

            blend_hbi = n.links.loc[blend_i, "bus1"]
            new_pnom_max = blend_hbi.map(pnom_by_hbi).fillna(0) * conversion
            n.links.loc[blend_i, "p_nom_max"] = new_pnom_max.values

            logger.info(
                f"Gas DRI H2 blend: set p_nom_max for {len(blend_i)} links "
                f"(total {new_pnom_max.sum():.1f} MW_gas from "
                f"{pnom_by_hbi.sum():.1f} MW_gas available gas DRI "
                f"(after deducting already-H2-retrofitted), "
                f"conversion={conversion:.4f})"
            )
        else:
            logger.info(
                "Gas DRI H2 blend: no existing gas DRI from previous periods; "
                "p_nom_max remains 0."
            )


def disable_grid_expansion_if_limit_hit(n):
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.
    """
    types = {"expansion_cost": "capital_cost", "volume_expansion": "length"}
    for limit_type in types:
        glcs = n.global_constraints.query(f"type == 'transmission_{limit_type}_limit'")

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {types[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {types[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)


def adjust_renewable_profiles(n, input_profiles, params, year):
    """
    Adjusts renewable profiles according to the renewable technology specified,
    using the latest year below or equal to the selected year.
    """

    # temporal clustering
    dr = get_snapshots(params["snapshots"], params["drop_leap_day"])
    snapshotmaps = (
        pd.Series(dr, index=dr).where(lambda x: x.isin(n.snapshots), pd.NA).ffill()
    )

    for carrier in params["carriers"]:
        if carrier == "hydro":
            continue

        with xr.open_dataset(getattr(input_profiles, "profile_" + carrier)) as ds:
            if ds.indexes["bus"].empty or "year" not in ds.indexes:
                continue

            ds = ds.stack(bus_bin=["bus", "bin"])

            closest_year = max(
                (y for y in ds.year.values if y <= year), default=min(ds.year.values)
            )

            p_max_pu = ds["profile"].sel(year=closest_year).to_pandas()
            p_max_pu.columns = p_max_pu.columns.map(flatten) + f" {carrier}"

            # temporal_clustering
            p_max_pu = p_max_pu.groupby(snapshotmaps).mean()

            # replace renewable time series
            n.generators_t.p_max_pu.loc[:, p_max_pu.columns] = p_max_pu


def update_heat_pump_efficiency(n: pypsa.Network, n_p: pypsa.Network, year: int):
    """
    Update the efficiency of heat pumps from previous years to current year
    (e.g. 2030 heat pumps receive 2040 heat pump COPs in 2030).

    Parameters
    ----------
    n : pypsa.Network
        The original network.
    n_p : pypsa.Network
        The network with the updated parameters.
    year : int
        The year for which the efficiency is being updated.

    Returns
    -------
    None
        This function updates the efficiency in place and does not return a value.
    """

    # get names of heat pumps in previous iteration that cannot be replaced by direct utilisation in this iteration
    heat_pump_idx_previous_iteration = n_p.links.index[
        n_p.links.index.str.contains("heat pump")
        & n_p.links.index.str[:-4].isin(
            n.links_t.efficiency.columns.str.rstrip(  # sources that can be directly used are no longer represented by heat pumps in the dynamic efficiency dataframe
                str(year)
            )
        )
    ]
    # construct names of same-technology heat pumps in the current iteration
    corresponding_idx_this_iteration = heat_pump_idx_previous_iteration.str[:-4] + str(
        year
    )
    # update efficiency of heat pumps in previous iteration in-place to efficiency in this iteration
    n_p.links_t["efficiency"].loc[:, heat_pump_idx_previous_iteration] = (
        n.links_t["efficiency"].loc[:, corresponding_idx_this_iteration].values
    )

    # Change efficiency2 for heat pumps that use an explicitly modelled heat source
    previous_iteration_columns = heat_pump_idx_previous_iteration.intersection(
        n_p.links_t["efficiency2"].columns
    )
    current_iteration_columns = corresponding_idx_this_iteration.intersection(
        n.links_t["efficiency2"].columns
    )
    n_p.links_t["efficiency2"].loc[:, previous_iteration_columns] = (
        n.links_t["efficiency2"].loc[:, current_iteration_columns].values
    )


def update_dynamic_ptes_capacity(
    n: pypsa.Network, n_p: pypsa.Network, year: int
) -> None:
    """
    Updates dynamic pit storage capacity based on district heating temperature changes.

    Parameters
    ----------
    n : pypsa.Network
        Original network.
    n_p : pypsa.Network
        Network with updated parameters.
    year : int
        Target year for capacity update.

    Returns
    -------
    None
        Updates capacity in-place.
    """
    # pit storages in previous iteration
    dynamic_ptes_idx_previous_iteration = n_p.stores.index[
        n_p.stores.index.str.contains("water pits")
    ]
    # construct names of same-technology dynamic pit storage in the current iteration
    corresponding_idx_this_iteration = dynamic_ptes_idx_previous_iteration.str[
        :-4
    ] + str(year)
    # update pit storage capacity in previous iteration in-place to capacity in this iteration
    n_p.stores_t.e_max_pu[dynamic_ptes_idx_previous_iteration] = n.stores_t.e_max_pu[
        corresponding_idx_this_iteration
    ].values


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            clusters="39",
            opts="",
            sector_opts="",
            planning_horizons=2050,
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)

    update_heat_pump_efficiency(n, n_p, year)

    if snakemake.params.tes and snakemake.params.dynamic_ptes_capacity:
        update_dynamic_ptes_capacity(n, n_p, year)

    add_brownfield(
        n,
        n_p,
        year,
        h2_retrofit=snakemake.params.H2_retrofit,
        h2_retrofit_capacity_per_ch4=snakemake.params.H2_retrofit_capacity_per_CH4,
        capacity_threshold=snakemake.params.threshold_capacity,
    )

    disable_grid_expansion_if_limit_hit(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    sanitize_custom_columns(n)
    sanitize_carriers(n, snakemake.config)
    n.export_to_netcdf(snakemake.output[0])
