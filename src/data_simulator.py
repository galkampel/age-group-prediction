import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_neighborhoods: int = 20, num_apartments: int = 10000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic neighborhood-level and apartment-level data.

    Parameters
    ----------
    num_neighborhoods : int, default 20
        Number of neighborhoods to generate.
    num_apartments : int, default 10000
        Total number of apartments to distribute across neighborhoods.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    df_neighborhoods : pd.DataFrame
        DataFrame representing neighborhood-level features.
    df_apartments : pd.DataFrame
        DataFrame representing apartment-level features.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate Neighborhoods
    neighborhood_ids = [f"N_{i+1:02d}" for i in range(num_neighborhoods)]

    # Generate parent/neighborhood-level characteristics
    # - avg_family_size: between 1.5 and 4.5
    # - cnt_schools: between 0 and 5
    # - cnt_parks: between 0 and 10
    avg_family_sizes = rng.uniform(1.5, 4.5, size=num_neighborhoods)
    cnt_schools = rng.integers(0, 6, size=num_neighborhoods)
    cnt_parks = rng.integers(0, 11, size=num_neighborhoods)

    df_neighborhoods = pd.DataFrame(
        {
            "neighborhood": neighborhood_ids,
            "avg_family_size": avg_family_sizes,
            "cnt_schools": cnt_schools,
            "cnt_parks": cnt_parks,
        }
    )

    # 2. Generate Apartments
    # Distribute the 10,000 apartments across neighborhoods.
    # We can assign weights or just distribute uniformly or with slight variance.
    # Let's do a dirichlet-based distribution of sizes or multinomial to make it realistic.
    alpha = rng.uniform(2.0, 10.0, size=num_neighborhoods)
    weights = rng.dirichlet(alpha)
    apartments_per_neighborhood = rng.multinomial(num_apartments, weights)

    # In case there's any discrepancy, ensure they sum to num_apartments
    # and adjust if necessary, though multinomial is guaranteed to sum to num_apartments.

    apartment_records = []
    # Create apartment-level features and relate them to the neighborhood context
    for i, neighborhood_id in enumerate(neighborhood_ids):
        count = apartments_per_neighborhood[i]
        if count == 0:
            continue

        # Get parent neighborhood details to influence child (apartment) distributions
        fam_size = avg_family_sizes[i]

        # cnt_rooms: logically correlated with average family size in that neighborhood
        # Assume Poisson or normal distribution rounded, with lower bound of 1.
        cnt_rooms = rng.poisson(lam=fam_size, size=count)
        cnt_rooms = np.clip(cnt_rooms, 1, 10)  # at least 1 room, max 10

        # surface: correlated with the number of rooms (e.g., ~15-25 sqm per room + some noise)
        base_surface = cnt_rooms * 20.0
        surface = base_surface + rng.normal(loc=10.0, scale=15.0, size=count)
        surface = np.clip(surface, 15.0, 300.0)  # realistic limits

        # Neighborhood infrastructure might correlate with local apartment conveniences
        # For example, count of kindergarten/elementary/highschool nearby might be locally distributed.
        # Let's generate cnt_kindergarten, cnt_elementary, cnt_highschool
        # correlated with local neighborhood stats or simply randomly with neighborhood constraint properties.
        n_schools = cnt_schools[i]

        # Generate local kindergarten counts per apartment (e.g. Poisson)
        cnt_kindergarten = rng.poisson(lam=1.2, size=count)
        # Elementary and Highschool counts nearby could be proportional to neighborhood's school count
        ele_lam = max(0.5, n_schools * 0.5)
        high_lam = max(0.3, n_schools * 0.3)

        cnt_elementary = rng.poisson(lam=ele_lam, size=count)
        cnt_highschool = rng.poisson(lam=high_lam, size=count)

        for idx in range(count):
            apartment_records.append(
                {
                    "cnt_rooms": int(cnt_rooms[idx]),
                    "surface": float(surface[idx]),
                    "neighborhood": neighborhood_id,
                    "cnt_kindergarten": int(cnt_kindergarten[idx]),
                    "cnt_elementary": int(cnt_elementary[idx]),
                    "cnt_highschool": int(cnt_highschool[idx]),
                }
            )

    df_apartments = pd.DataFrame(apartment_records)
   
    return df_neighborhoods, df_apartments
