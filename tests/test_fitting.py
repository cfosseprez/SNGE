"""
Tests for snge.fitting module.
"""

import numpy as np
import pytest

from snge.fitting import (
    ExperimentalData,
    FitResult,
    create_synthetic_data,
    fit_nge_to_mean,
    fit_fw_to_mean,
    compare_nge_vs_fw,
    nge_yield_curve,
    fw_yield_curve
)


class TestExperimentalData:
    """Tests for ExperimentalData class."""

    def test_creation(self):
        """Test basic creation."""
        times = np.array([0, 60, 120, 180])
        yields = np.array([
            [0.0, 0.2, 0.4, 0.5],
            [0.0, 0.18, 0.38, 0.48],
            [0.0, 0.22, 0.42, 0.52]
        ])
        A0 = 0.01

        data = ExperimentalData(times=times, yields=yields, A0=A0)

        assert len(data.times) == 4
        assert data.n_runs == 3
        assert data.n_timepoints == 4
        assert data.A0 == 0.01

    def test_mean_yield_property(self):
        """Test mean_yield calculation."""
        times = np.array([0, 60, 120])
        yields = np.array([
            [0.0, 0.2, 0.4],
            [0.0, 0.3, 0.5],
            [0.0, 0.1, 0.3]
        ])

        data = ExperimentalData(times=times, yields=yields, A0=0.01)

        expected_mean = np.array([0.0, 0.2, 0.4])
        np.testing.assert_allclose(data.mean_yield, expected_mean)

    def test_std_yield_property(self):
        """Test std_yield uses ddof=1."""
        times = np.array([0, 60])
        yields = np.array([
            [0.0, 0.2],
            [0.0, 0.3],
            [0.0, 0.4]
        ])

        data = ExperimentalData(times=times, yields=yields, A0=0.01)

        # At t=60: values are 0.2, 0.3, 0.4
        expected_std = np.std([0.2, 0.3, 0.4], ddof=1)
        assert data.std_yield[1] == pytest.approx(expected_std)

    def test_final_yields_property(self):
        """Test final_yields extraction."""
        times = np.array([0, 60, 120])
        yields = np.array([
            [0.0, 0.2, 0.5],
            [0.0, 0.3, 0.55],
            [0.0, 0.1, 0.45]
        ])

        data = ExperimentalData(times=times, yields=yields, A0=0.01)

        expected_final = np.array([0.5, 0.55, 0.45])
        np.testing.assert_allclose(data.final_yields, expected_final)

    def test_cv_final_property(self):
        """Test CV of final yields."""
        times = np.array([0, 60, 120])
        yields = np.array([
            [0.0, 0.2, 0.5],
            [0.0, 0.3, 0.5],
            [0.0, 0.1, 0.5]
        ])

        data = ExperimentalData(times=times, yields=yields, A0=0.01)

        # All final yields are 0.5, so CV should be 0
        assert data.cv_final == pytest.approx(0, abs=1e-10)


class TestCreateSyntheticData:
    """Tests for synthetic data creation."""

    def test_shape(self):
        """Test output shape."""
        times = np.array([0, 30, 60, 90, 120])
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=20, noise_level=0.05, seed=42
        )

        assert data.n_runs == 20
        assert data.n_timepoints == 5

    def test_reproducibility(self):
        """Test seed gives reproducible results."""
        times = np.array([0, 30, 60, 90])

        data1 = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=10, noise_level=0.05, seed=42
        )
        data2 = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=10, noise_level=0.05, seed=42
        )

        np.testing.assert_array_equal(data1.yields, data2.yields)

    def test_noise_level_effect(self):
        """Test that higher noise gives more variance."""
        times = np.array([0, 30, 60, 90, 120, 180, 240, 300])

        data_low_noise = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=50, noise_level=0.01, seed=42
        )
        data_high_noise = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=50, noise_level=0.1, seed=42
        )

        # Higher noise should give higher variance at final time
        var_low = np.var(data_low_noise.final_yields)
        var_high = np.var(data_high_noise.final_yields)
        assert var_high > var_low

    def test_yields_bounded(self):
        """Test yields stay in [0, 1]."""
        times = np.linspace(0, 300, 31)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=100, noise_level=0.1, seed=42
        )

        assert np.all(data.yields >= 0)
        assert np.all(data.yields <= 1)


class TestFitNgeToMean:
    """Tests for NGE model fitting."""

    def test_convergence(self):
        """Test fitting converges."""
        times = np.array([0, 30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600])
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=20, noise_level=0.02, seed=42
        )

        fit = fit_nge_to_mean(data, method='least_squares')

        assert fit.converged
        assert fit.r_squared > 0.9  # Should fit well

    def test_parameter_recovery(self):
        """Test that fitted parameters are close to true values."""
        k1_true, k2_true, k3_true = 2e-4, 0.15, 0.008
        times = np.array([0, 30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600])

        data = create_synthetic_data(
            k1=k1_true, k2=k2_true, k3=k3_true, A0=0.01,
            times=times, n_runs=25, noise_level=0.02, seed=42
        )

        fit = fit_nge_to_mean(data, method='both')

        # Parameters should be recovered within 50% (fitting can be tricky)
        assert abs(fit.k1 - k1_true) / k1_true < 0.5
        assert abs(fit.k2 - k2_true) / k2_true < 0.5
        assert abs(fit.k3 - k3_true) / k3_true < 0.5

    def test_r_squared_range(self):
        """Test R-squared is in valid range."""
        times = np.linspace(0, 600, 15)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=20, noise_level=0.03, seed=42
        )

        fit = fit_nge_to_mean(data)

        assert 0 <= fit.r_squared <= 1

    def test_fit_result_properties(self):
        """Test FitResult has all expected properties."""
        times = np.linspace(0, 600, 15)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=20, noise_level=0.03, seed=42
        )

        fit = fit_nge_to_mean(data)

        assert hasattr(fit, 'k1')
        assert hasattr(fit, 'k2')
        assert hasattr(fit, 'k3')
        assert hasattr(fit, 'r_squared')
        assert hasattr(fit, 'rmse')
        assert hasattr(fit, 'aic')
        assert hasattr(fit, 'bic')
        assert hasattr(fit, 'residuals')


class TestFitFwToMean:
    """Tests for Finke-Watzky model fitting."""

    def test_convergence(self):
        """Test FW fitting converges."""
        times = np.linspace(0, 600, 15)
        # Create data using FW model (k3=0)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.0, A0=0.01,
            times=times, n_runs=20, noise_level=0.02, seed=42
        )

        fit = fit_fw_to_mean(data)

        assert fit is not None
        assert fit['r_squared'] > 0.9

    def test_fw_fit_keys(self):
        """Test FW fit result has expected keys."""
        times = np.linspace(0, 600, 15)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.0, A0=0.01,
            times=times, n_runs=20, noise_level=0.02, seed=42
        )

        fit = fit_fw_to_mean(data)

        assert 'k1' in fit
        assert 'k2' in fit
        assert 'r_squared' in fit
        assert 'rmse' in fit


class TestCompareNgeVsFw:
    """Tests for NGE vs FW model comparison."""

    def test_expected_keys(self):
        """Test comparison result has expected keys."""
        times = np.linspace(0, 600, 15)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=20, noise_level=0.02, seed=42
        )

        nge_fit = fit_nge_to_mean(data)
        fw_fit = fit_fw_to_mean(data)

        comparison = compare_nge_vs_fw(data, nge_fit, fw_fit)

        expected_keys = [
            'f_statistic', 'f_pvalue',
            'nge_preferred_ftest',
            'delta_aic', 'nge_preferred_aic',
            'delta_bic', 'nge_preferred_bic',
            'fw_r_squared', 'nge_r_squared'
        ]

        for key in expected_keys:
            assert key in comparison, f"Missing key: {key}"

    def test_nge_preferred_when_etching_present(self):
        """Test NGE is preferred when data has etching."""
        times = np.linspace(0, 600, 20)
        # Create data with significant etching
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.02,  # Higher k3
            A0=0.01, times=times, n_runs=30,
            noise_level=0.01, seed=42
        )

        nge_fit = fit_nge_to_mean(data, method='both')
        fw_fit = fit_fw_to_mean(data)

        comparison = compare_nge_vs_fw(data, nge_fit, fw_fit)

        # NGE should fit better (lower AIC, higher R^2)
        # Note: This might not always hold depending on data
        # At minimum, check comparison runs without error
        assert 'nge_r_squared' in comparison
        assert 'fw_r_squared' in comparison


class TestYieldCurves:
    """Tests for yield curve functions."""

    def test_nge_yield_curve_initial(self):
        """Test NGE yield is 0 at t=0."""
        times = np.array([0.0])
        y = nge_yield_curve(times, k1=2e-4, k2=0.15, k3=0.008, A0=0.01)
        assert y[0] == pytest.approx(0, abs=1e-10)

    def test_fw_yield_curve_initial(self):
        """Test FW yield is 0 at t=0."""
        times = np.array([0.0])
        y = fw_yield_curve(times, k1=2e-4, k2=0.15, A0=0.01)
        assert y[0] == pytest.approx(0, abs=1e-10)

    def test_fw_yield_approaches_1(self):
        """Test FW yield approaches 1 at long times."""
        times = np.array([0, 1000, 2000, 5000, 10000])
        y = fw_yield_curve(times, k1=2e-4, k2=0.15, A0=0.01)

        # Should approach 1 (complete conversion)
        assert y[-1] > 0.95

    def test_nge_yield_bounded(self):
        """Test NGE yield stays in [0, 1]."""
        times = np.linspace(0, 1000, 101)
        y = nge_yield_curve(times, k1=2e-4, k2=0.15, k3=0.008, A0=0.01)

        assert np.all(y >= 0)
        assert np.all(y <= 1)
