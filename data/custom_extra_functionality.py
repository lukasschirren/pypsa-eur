# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


def _add_ua_nuclear_average_cf_constraint(n, snakemake):
    """
    Limit total annual energy output of UA nuclear Links to an average capacity
    factor, i.e.:
        sum_t( p[link,t] * w[t] ) <= cf * T * p_nom[link]   for each UA nuclear link

    where p_nom is fixed for brownfield links or a variable for extendable links.
    Parameters are read from config conventional.nuclear.average_max_cf_pu.
    """
    nuclear_params = snakemake.config.get("conventional", {}).get("nuclear", {})
    avg_max_cf = nuclear_params.get("average_max_cf_pu", {})

    if not avg_max_cf or "UA" not in avg_max_cf:
        return

    cf = avg_max_cf["UA"]

    # Find all UA nuclear Links (both brownfield vintages and new-build extendable)
    ua_nuclear = n.links[
        (n.links.carrier == "nuclear")
        & (n.links.bus1.str.startswith("UA"))
    ].index

    if ua_nuclear.empty:
        return

    w = n.snapshot_weightings["generators"]
    T = float(w.sum())

    # Weighted dispatch sum for all UA nuclear links (xarray linear expression)
    dispatch = n.model["Link-p"].loc[:, ua_nuclear]
    total_energy = (dispatch * w).sum()

    # Split into fixed-p_nom (brownfield) and extendable
    ua_fixed = ua_nuclear[~n.links.loc[ua_nuclear, "p_nom_extendable"]]
    ua_ext = ua_nuclear[n.links.loc[ua_nuclear, "p_nom_extendable"]]

    rhs = cf * T * float(n.links.loc[ua_fixed, "p_nom"].sum()) if len(ua_fixed) else 0.0

    if len(ua_ext):
        # For extendable links p_nom is a variable; move to LHS
        p_nom_ext = n.model["Link-p_nom"].loc[ua_ext]
        lhs = total_energy - cf * T * p_nom_ext.sum()
        n.model.add_constraints(lhs <= rhs, name="ua_nuclear_avg_cf_max")
    else:
        n.model.add_constraints(total_energy <= rhs, name="ua_nuclear_avg_cf_max")

    import logging
    logging.getLogger(__name__).info(
        f"Added ua_nuclear_avg_cf_max constraint: cf={cf}, T={T:.0f}h, "
        f"fixed_cap={n.links.loc[ua_fixed, 'p_nom'].sum():.0f} MW, "
        f"extendable_links={len(ua_ext)}"
    )


def custom_extra_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    _add_ua_nuclear_average_cf_constraint(n, snakemake)
