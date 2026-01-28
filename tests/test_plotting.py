"""
Tests for snge.plotting module.

These are smoke tests that verify plotting functions don't raise exceptions.
Actual visual correctness would require manual inspection or image comparison.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from snge.plotting import (
    plot_ensemble_trajectories,
    plot_cv_evolution,
    plot_distribution_comparison,
    plot_summary_figure
)
from snge.models import NGEParameters, SimulationResult


@pytest.fixture
def plotting_params():
    """Parameters for plotting tests."""
    return NGEParameters(
        k1=2e-4, k2=0.15, k3=0.008,
        A0=0.01, B0=0.0, V=1e-6, t_max=100.0
    )


@pytest.fixture
def mock_results_for_plotting(plotting_params):
    """Create mock results for plotting tests."""
    np.random.seed(42)
    n_runs = 20
    n_points = 101
    times = np.linspace(0, plotting_params.t_max, n_points)

    results = []
    for i in range(n_runs):
        base_yield = 0.6 + 0.1 * np.random.randn()
        B_conc = plotting_params.A0 * base_yield * (1 - np.exp(-0.03 * times))
        A_conc = plotting_params.A0 - B_conc

        results.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=B_conc[-1] / plotting_params.A0,
            method="Mock"
        ))

    return results


class TestPlotEnsembleTrajectories:
    """Tests for ensemble trajectory plotting."""

    def test_returns_figure(self, mock_results_for_plotting, plotting_params):
        """Test that function returns a Figure."""
        fig = plot_ensemble_trajectories(
            mock_results_for_plotting, plotting_params, n_show=10
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_exceptions(self, mock_results_for_plotting, plotting_params):
        """Test plotting doesn't raise exceptions."""
        try:
            fig = plot_ensemble_trajectories(
                mock_results_for_plotting, plotting_params
            )
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_ensemble_trajectories raised {e}")

    def test_custom_figsize(self, mock_results_for_plotting, plotting_params):
        """Test custom figure size."""
        fig = plot_ensemble_trajectories(
            mock_results_for_plotting, plotting_params,
            figsize=(8, 4)
        )

        # Check figure size (in inches)
        assert fig.get_figwidth() == pytest.approx(8)
        assert fig.get_figheight() == pytest.approx(4)
        plt.close(fig)


class TestPlotCvEvolution:
    """Tests for CV evolution plotting."""

    def test_returns_figure(self, mock_results_for_plotting, plotting_params):
        """Test that function returns a Figure."""
        fig = plot_cv_evolution(mock_results_for_plotting, plotting_params)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_exceptions(self, mock_results_for_plotting, plotting_params):
        """Test plotting doesn't raise exceptions."""
        try:
            fig = plot_cv_evolution(mock_results_for_plotting, plotting_params)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_cv_evolution raised {e}")


class TestPlotDistributionComparison:
    """Tests for distribution comparison plotting."""

    def test_returns_figure(self, mock_results_for_plotting):
        """Test that function returns a Figure."""
        # Create two sets of results
        results_g = mock_results_for_plotting
        results_em = mock_results_for_plotting[:10]  # Use subset

        fig = plot_distribution_comparison(results_g, results_em)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_exceptions(self, mock_results_for_plotting):
        """Test plotting doesn't raise exceptions."""
        results_g = mock_results_for_plotting
        results_em = mock_results_for_plotting

        try:
            fig = plot_distribution_comparison(results_g, results_em)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_distribution_comparison raised {e}")

    def test_with_experimental_yields(self, mock_results_for_plotting):
        """Test with experimental yields provided."""
        results_g = mock_results_for_plotting
        results_em = mock_results_for_plotting
        exp_yields = np.array([0.55, 0.60, 0.58, 0.62, 0.57])

        try:
            fig = plot_distribution_comparison(
                results_g, results_em, experimental_yields=exp_yields
            )
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_distribution_comparison with exp yields raised {e}")


class TestPlotSummaryFigure:
    """Tests for summary figure plotting."""

    def test_returns_figure(self, mock_results_for_plotting, plotting_params):
        """Test that function returns a Figure."""
        fig = plot_summary_figure(mock_results_for_plotting, plotting_params)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_exceptions(self, mock_results_for_plotting, plotting_params):
        """Test plotting doesn't raise exceptions."""
        try:
            fig = plot_summary_figure(mock_results_for_plotting, plotting_params)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_summary_figure raised {e}")

    def test_with_experimental_yields(self, mock_results_for_plotting, plotting_params):
        """Test with experimental yields."""
        exp_yields = np.array([0.55, 0.60, 0.58, 0.62, 0.57])

        try:
            fig = plot_summary_figure(
                mock_results_for_plotting, plotting_params,
                experimental_yields=exp_yields
            )
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_summary_figure with exp yields raised {e}")

    def test_custom_figsize(self, mock_results_for_plotting, plotting_params):
        """Test custom figure size."""
        fig = plot_summary_figure(
            mock_results_for_plotting, plotting_params,
            figsize=(16, 12)
        )

        assert fig.get_figwidth() == pytest.approx(16)
        assert fig.get_figheight() == pytest.approx(12)
        plt.close(fig)


class TestPlottingCleanup:
    """Tests to ensure proper cleanup after plotting."""

    def test_close_figures(self, mock_results_for_plotting, plotting_params):
        """Test that figures can be properly closed."""
        fig1 = plot_ensemble_trajectories(
            mock_results_for_plotting, plotting_params
        )
        fig2 = plot_cv_evolution(mock_results_for_plotting, plotting_params)

        plt.close(fig1)
        plt.close(fig2)

        # No assertion - just verify no exceptions

    def test_multiple_plots_no_memory_leak(self, mock_results_for_plotting, plotting_params):
        """Test creating multiple plots doesn't cause issues."""
        for _ in range(5):
            fig = plot_ensemble_trajectories(
                mock_results_for_plotting, plotting_params, n_show=5
            )
            plt.close(fig)

        # Close all remaining figures
        plt.close('all')
