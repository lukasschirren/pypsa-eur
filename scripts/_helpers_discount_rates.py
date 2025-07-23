# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Helper functions for country-specific discount rates.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_country_discount_rate(country_code: str, costs_config: dict) -> float:
    """
    Get the discount rate for a specific country.
    
    Falls back to global social_discountrate if country-specific rate is not defined.
    
    Parameters
    ----------
    country_code : str
        ISO 2-letter country code (e.g., 'UA', 'DE')
    costs_config : dict
        Costs configuration from config file
        
    Returns
    -------
    float
        Discount rate for the country
    """
    country_rates = costs_config.get("country_specific_discountrate", {})
    
    if country_code in country_rates:
        rate = country_rates[country_code]
        logger.debug(f"Using country-specific discount rate for {country_code}: {rate:.1%}")
        return rate
    else:
        # Fallback to global discount rate if available
        global_rate = costs_config.get("social_discountrate")
        if global_rate is not None:
            rate = global_rate
            logger.debug(f"Using global discount rate for {country_code}: {rate:.1%}")
            return rate
        else:
            # Final fallback: use a standard discount rate of 7%
            rate = 0.07
            logger.warning(f"No discount rate specified in config. Using default rate of {rate:.1%} for {country_code}")
            return rate


def get_country_discount_rates_series(countries: list, costs_config: dict) -> pd.Series:
    """
    Get discount rates for multiple countries as a pandas Series.
    
    Parameters
    ----------
    countries : list
        List of ISO 2-letter country codes
    costs_config : dict
        Costs configuration from config file
        
    Returns
    -------
    pd.Series
        Series with countries as index and discount rates as values
    """
    discount_rates = {}
    for country in countries:
        discount_rates[country] = get_country_discount_rate(country, costs_config)
    
    return pd.Series(discount_rates, name="discount_rate")


def get_nodal_discount_rates(network_buses: pd.DataFrame, costs_config: dict) -> pd.Series:
    """
    Get discount rates for network buses based on their country.
    
    Parameters
    ----------
    network_buses : pd.DataFrame
        Network buses DataFrame with 'country' column
    costs_config : dict
        Costs configuration from config file
        
    Returns
    -------
    pd.Series
        Series with bus names as index and discount rates as values
    """
    # Extract country codes from bus names or use country column if available
    if 'country' in network_buses.columns:
        countries = network_buses['country']
    else:
        # Extract country code from bus names (first 2 characters)
        countries = network_buses.index.str[:2]
    
    discount_rates = {}
    for bus_id, country in countries.items():
        discount_rates[bus_id] = get_country_discount_rate(country, costs_config)
    
    return pd.Series(discount_rates, name="discount_rate")


def apply_country_specific_costs(costs: pd.DataFrame, countries: list, costs_config: dict) -> pd.DataFrame:
    """
    Apply country-specific discount rates to technology costs.
    
    This function modifies the 'discount rate' column in the costs DataFrame
    to enable country-specific calculations.
    
    Parameters
    ----------
    costs : pd.DataFrame
        Technology costs DataFrame
    countries : list
        List of countries to consider
    costs_config : dict
        Costs configuration from config file
        
    Returns
    -------
    pd.DataFrame
        Modified costs DataFrame with country-specific structure
    """
    # Create a multi-index costs DataFrame with countries
    if len(countries) > 1:
        # Create multi-index for (technology, country)
        country_costs = []
        for country in countries:
            country_df = costs.copy()
            country_df['discount rate'] = get_country_discount_rate(country, costs_config)
            country_df['country'] = country
            country_costs.append(country_df)
        
        # Concatenate and set multi-index
        expanded_costs = pd.concat(country_costs, keys=countries)
        expanded_costs.index.names = ['country', 'technology']
        
        return expanded_costs
    else:
        # Single country case
        country = countries[0]
        costs['discount rate'] = get_country_discount_rate(country, costs_config)
        return costs


def apply_country_discount_rates_to_network(network, costs_config: dict) -> None:
    """
    Apply country-specific discount rates to network components by recalculating capital costs.
    
    This function recalculates capital costs for network components using country-specific 
    discount rates extracted from bus locations. It requires original investment costs
    and technology data to properly recalculate annuities.
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with components
    costs_config : dict
        Costs configuration containing country-specific discount rates
    """
    logger.info("Applying country-specific discount rates to network components...")
    
    # Get country-specific discount rates from config
    country_rates = costs_config.get("country_specific_discountrate", {})
    global_rate = costs_config["social_discountrate"]
    
    if not country_rates:
        logger.info("No country-specific discount rates found, using global rate.")
        return
    
    # We need access to the original costs data to recalculate properly
    # For now, we'll apply a simplified approach using rate ratios
    # A more robust implementation would require reloading technology costs
    
    components_to_process = []
    
    # Process generators
    if hasattr(network, 'generators') and not network.generators.empty:
        components_to_process.append(('generators', network.generators))
    
    # Process storage units
    if hasattr(network, 'storage_units') and not network.storage_units.empty:
        components_to_process.append(('storage_units', network.storage_units))
    
    # Process stores
    if hasattr(network, 'stores') and not network.stores.empty:
        components_to_process.append(('stores', network.stores))
    
    # Process links
    if hasattr(network, 'links') and not network.links.empty:
        components_to_process.append(('links', network.links))
    
    total_adjustments = 0
    
    for component_name, component in components_to_process:
        adjustments = _apply_to_component_improved(
            component, component_name, country_rates, global_rate
        )
        total_adjustments += adjustments
    
    if total_adjustments > 0:
        logger.info(f"Applied country-specific discount rates to {total_adjustments} components total")
    else:
        logger.info("No components required discount rate adjustments")


def _apply_to_component_improved(component, component_name: str, 
                                country_rates: dict, global_rate: float) -> int:
    """
    Apply country-specific discount rates to a specific component type (improved version).
    
    Parameters
    ----------
    component : pd.DataFrame
        Component DataFrame (generators, storage_units, etc.)
    component_name : str
        Name of component ('generators', 'storage_units', etc.)
    country_rates : dict
        Country-specific discount rates
    global_rate : float
        Global discount rate
        
    Returns
    -------
    int
        Number of components that were adjusted
    """
    if component.empty:
        return 0
    
    # Extract country codes from bus names or use bus country info
    if 'bus' in component.columns:
        countries = component['bus'].str[:2]
    elif 'country' in component.columns:
        countries = component['country']
    else:
        # Try to extract from index (component names like "DE0 onwind")
        countries = component.index.str[:2]
    
    # Check if we have any capital costs to adjust
    if 'capital_cost' not in component.columns:
        return 0
    
    # Apply country-specific rates using annuity recalculation approach
    changed_components = []
    
    # Iterate over the component index and corresponding countries
    for i, idx in enumerate(component.index):
        # Get the country for this component
        if hasattr(countries, 'loc'):
            country = countries.loc[idx]
        elif hasattr(countries, 'iloc'):
            country = countries.iloc[i]
        else:
            # Fallback for Index objects
            country = countries[i]
            
        if country in country_rates:
            country_rate = country_rates[country]
            if abs(country_rate - global_rate) > 0.001:  # Only adjust if significantly different
                
                # For a proper implementation, we would need:
                # 1. Original investment cost
                # 2. Technology lifetime 
                # 3. FOM rate
                # Since these are not readily available in the network components,
                # we use a simplified approach based on discount rate ratios
                
                # Calculate annuity factor ratio (approximate adjustment)
                # This assumes a typical lifetime (25 years) for the adjustment
                # Real implementation should use actual technology lifetimes
                default_lifetime = 25
                old_annuity = calculate_annuity_simple(default_lifetime, global_rate)
                new_annuity = calculate_annuity_simple(default_lifetime, country_rate)
                adjustment_factor = new_annuity / old_annuity
                
                # Apply the adjustment
                component.loc[idx, 'capital_cost'] *= adjustment_factor
                changed_components.append((idx, country, country_rate, adjustment_factor))
    
    if changed_components:
        logger.info(f"Applied country-specific rates to {len(changed_components)} {component_name}")
        for comp_id, country, rate, factor in changed_components[:3]:  # Log first 3
            logger.debug(f"  {comp_id} ({country}): {rate:.1%} (factor: {factor:.3f})")
        if len(changed_components) > 3:
            logger.debug(f"  ... and {len(changed_components) - 3} more")
    
    return len(changed_components)


def calculate_annuity_simple(lifetime: float, discount_rate: float) -> float:
    """
    Calculate simple annuity factor.
    
    Parameters
    ----------
    lifetime : float
        Asset lifetime in years
    discount_rate : float
        Discount rate
        
    Returns
    -------
    float
        Annuity factor
    """
    if discount_rate == 0:
        return 1 / lifetime
    return discount_rate / (1 - (1 + discount_rate) ** (-lifetime))
