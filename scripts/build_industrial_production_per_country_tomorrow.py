# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build future industrial production per country.

Description
-------

This rule uses the ``industrial_production_per_country.csv`` file and the expected recycling rates to calculate the future production of the industrial sectors.

**St_primary_fraction**
The fraction of steel that is coming from primary production. This is more energy intensive than recycling steel (secondary production).

**DRI_fraction**
The fraction of primary steel that is produced in DRI plants.

**Al_primary_fraction**
The fraction of aluminium that is coming from primary production. This is more energy intensive than recycling aluminium (secondary production).

**HVC_primary_fraction**
The fraction of high value chemicals that are coming from primary production (crude oil or Fischer Tropsch).

**HVC_mechanical_recycling_fraction**
The fraction of high value chemicals that are coming from mechanical recycling.

**HVC_chemical_recycling_fraction**
The fraction of high value chemicals that are coming from chemical recycling.

If not already present, the information is added as new column in the output file.

The unit of the production is kt/a.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.prepare_sector_network import get

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_industrial_production_per_country_tomorrow")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.industry

    investment_year = int(snakemake.wildcards.planning_horizons)

    fn = snakemake.input.industrial_production_per_country
    production = pd.read_csv(fn, index_col=0)

    keys = ["Integrated steelworks", "Electric arc"]
    total_steel = production[keys].sum(axis=1)

    st_primary_fraction = get(params["St_primary_fraction"], investment_year)
    dri_fraction = get(params["DRI_fraction"], investment_year)
    int_steel = production["Integrated steelworks"].sum()
    fraction_persistent_primary = st_primary_fraction * total_steel.sum() / int_steel

    dri = (
        dri_fraction * fraction_persistent_primary * production["Integrated steelworks"]
    )
    production.insert(2, "DRI + Electric arc", dri)

    not_dri = 1 - dri_fraction
    production["Integrated steelworks"] = (
        not_dri * fraction_persistent_primary * production["Integrated steelworks"]
    )
    production["Electric arc"] = (
        total_steel
        - production["DRI + Electric arc"]
        - production["Integrated steelworks"]
    )

    # Ukraine-specific steel production override
    ua_steel_config = params.get("ua_steel_production", {})
    ua_primary_config = params.get("ua_St_primary_fraction", {})
    ua_dri_config = params.get("ua_DRI_fraction", {})
    
    if ua_steel_config or ua_primary_config or ua_dri_config:
        ua_mask = production.index.str.startswith("UA")
        
        # Get Ukraine-specific fractions (fall back to global if not specified)
        ua_st_primary = get(ua_primary_config, investment_year) if ua_primary_config else None
        if ua_st_primary is None:
            ua_st_primary = st_primary_fraction
        
        ua_dri = get(ua_dri_config, investment_year) if ua_dri_config else None
        if ua_dri is None:
            ua_dri = dri_fraction
        
        # Apply Ukraine-specific primary/secondary and DRI split if different from global
        if ua_primary_config or ua_dri_config:
            ua_total_steel = production.loc[ua_mask, ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]].sum(axis=1)
            ua_primary_steel = ua_total_steel * ua_st_primary
            ua_secondary_steel = ua_total_steel * (1 - ua_st_primary)
            
            # Distribute primary between DRI and Integrated steelworks using Ukraine-specific DRI_fraction
            production.loc[ua_mask, "DRI + Electric arc"] = ua_primary_steel * ua_dri
            production.loc[ua_mask, "Integrated steelworks"] = ua_primary_steel * (1 - ua_dri)
            production.loc[ua_mask, "Electric arc"] = ua_secondary_steel
            
            logger.info(f"Applied Ukraine-specific fractions: primary={ua_st_primary:.1%}, DRI={ua_dri:.1%} for {investment_year}")
        
        # Then scale to match target demand if specified
        if ua_steel_config:
            ua_steel_demand = get(ua_steel_config, investment_year)
            if ua_steel_demand is not None:
                ua_rows = production.loc[ua_mask]
                ua_total = ua_rows[["Electric arc", "Integrated steelworks", "DRI + Electric arc"]].sum().sum()

                if ua_total > 0:
                    scale_factor = ua_steel_demand / ua_total
                    for col in ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]:
                        production.loc[ua_mask, col] *= scale_factor
                else:
                    # If no existing steel, distribute based on technology fractions
                    n_ua_nodes = ua_mask.sum()
                    per_node = ua_steel_demand / n_ua_nodes
                    production.loc[ua_mask, "DRI + Electric arc"] = per_node * ua_dri * ua_st_primary
                    production.loc[ua_mask, "Integrated steelworks"] = per_node * (1 - ua_dri) * ua_st_primary
                    production.loc[ua_mask, "Electric arc"] = per_node * (1 - ua_st_primary)

                logger.info(f"Set Ukraine steel production to {ua_steel_demand} kt/a for {investment_year}")

    keys = ["Aluminium - primary production", "Aluminium - secondary production"]
    total_aluminium = production[keys].sum(axis=1)

    key_pri = "Aluminium - primary production"
    key_sec = "Aluminium - secondary production"

    al_primary_fraction = get(params["Al_primary_fraction"], investment_year)
    fraction_persistent_primary = (
        al_primary_fraction * total_aluminium.sum() / (production[key_pri].sum() or 1)
    )

    production[key_pri] = fraction_persistent_primary * production[key_pri]
    production[key_sec] = total_aluminium - production[key_pri]

    production["HVC (mechanical recycling)"] = (
        get(params["HVC_mechanical_recycling_fraction"], investment_year)
        * production["HVC"]
    )
    production["HVC (chemical recycling)"] = (
        get(params["HVC_chemical_recycling_fraction"], investment_year)
        * production["HVC"]
    )

    production["HVC"] *= get(params["HVC_primary_fraction"], investment_year)

    fn = snakemake.output.industrial_production_per_country_tomorrow
    production.to_csv(fn, float_format="%.2f")
