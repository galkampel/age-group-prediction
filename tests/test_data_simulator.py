import pytest
import pandas as pd
from data_simulator import SyntheticDataSimulator


def test_generate_synthetic_data_structure():
    simulator = SyntheticDataSimulator(seed=42)
    df_n, df_a = simulator.create_data(num_neighborhoods=20, num_apartments=10000)
    
    # Check returns are DataFrames
    assert isinstance(df_n, pd.DataFrame)
    assert isinstance(df_a, pd.DataFrame)
    
    # Check sizes
    assert len(df_n) == 20
    assert len(df_a) == 10000
    
    # Check neighborhood columns
    expected_n_cols = {'neighborhood', 'avg_family_size', 'cnt_schools', 'cnt_parks'}
    assert expected_n_cols.issubset(df_n.columns)
    
    # Check apartment columns
    expected_a_cols = {'cnt_rooms', 'surface', 'neighborhood', 'cnt_kindergarten', 'cnt_elementary', 'cnt_highschool'}
    assert expected_a_cols.issubset(df_a.columns)


def test_consistency_and_seeds():
    simulator1 = SyntheticDataSimulator(seed=42)
    simulator2 = SyntheticDataSimulator(seed=42)
    simulator3 = SyntheticDataSimulator(seed=100)

    df_n1, df_a1 = simulator1.create_data(num_neighborhoods=20, num_apartments=1000)
    df_n2, df_a2 = simulator2.create_data(num_neighborhoods=20, num_apartments=1000)
    df_n3, df_a3 = simulator3.create_data(num_neighborhoods=20, num_apartments=1000)
    
    pd.testing.assert_frame_equal(df_n1, df_n2)
    pd.testing.assert_frame_equal(df_a1, df_a2)
    
    # Different seed should generate different data
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df_n1, df_n3)


def test_logical_rules():
    simulator = SyntheticDataSimulator(seed=42)
    df_n, df_a = simulator.create_data(num_neighborhoods=5, num_apartments=100)
    
    # Rooms should be at least 1
    assert (df_a['cnt_rooms'] >= 1).all()
    
    # Surfaces should be positive
    assert (df_a['surface'] > 0).all()
    
    # Kindergarten count should be non-negative
    assert (df_a['cnt_kindergarten'] >= 0).all()
    
    # Elementary count should be non-negative
    assert (df_a['cnt_elementary'] >= 0).all()
    
    # Highschool count should be non-negative
    assert (df_a['cnt_highschool'] >= 0).all()
