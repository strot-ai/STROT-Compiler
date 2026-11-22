import math
import random

import pytest

from strot.calibrate import (
    apply_temperature,
    calibration_report,
    conformal_quantile,
    fit_split_calibration,
    fit_temperature,
    fit_temperature_logits,
    prediction_set,
    softmax_logits,
)


class TestTemperature:
    def test_identity_at_t1(self):
        probs = [0.7, 0.2, 0.1]
        assert apply_temperature(probs, 1.0) == pytest.approx(probs)

    def test_high_t_flattens_low_t_sharpens(self):
        probs = [0.7, 0.2, 0.1]
        flat = apply_temperature(probs, 4.0)
        sharp = apply_temperature(probs, 0.25)
        assert flat[0] < probs[0] < sharp[0]
        assert sum(flat) == pytest.approx(1.0)
        assert sum(sharp) == pytest.approx(1.0)

    def test_logit_temperature_softens_wrong_sharpens_right(self):
        # Logit-space temperature (used by decide's per-question heads): a head
        # that is confidently WRONG half the time softens (T>1); a head that is
        # confidently RIGHT may sharpen (T<1). Saturated 0/1 probs cannot be
        # recovered by prob-space scaling, so decide fits on raw logits.
        sharp_mixed = [[6.0, -6.0], [6.0, -6.0], [-6.0, 6.0], [-6.0, 6.0]]
        assert fit_temperature_logits(sharp_mixed, [0, 1, 0, 1]) > 1.0  # half wrong
        assert fit_temperature_logits([[6.0, -6.0]] * 4, [0, 0, 0, 0]) < 1.0  # all right
        # Softening a sharp, wrong-leaning logit pair lifts the minority mass.
        assert softmax_logits([6.0, -6.0], 4.0)[1] > softmax_logits([6.0, -6.0], 1.0)[1]

    def test_logit_temperature_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_temperature_logits([[1.0, 2.0]], [0, 1])

    def test_fit_softens_overconfident_model(self):
        # Model says 0.9 but is right only ~half the time -> T should be > 1.
        distributions = [[0.9, 0.1]] * 10
        truths = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        assert fit_temperature(distributions, truths) > 1.0

    def test_fit_keeps_wellcalibrated_model_near_t1(self):
        distributions = [[0.9, 0.1]] * 10
        truths = [0] * 9 + [1]  # right 90% of the time at 0.9 confidence
        assert 0.6 <= fit_temperature(distributions, truths) <= 1.7


class TestConformal:
    def test_quantile_covers_easy_set(self):
        # Perfectly separable: true label always at 0.95.
        distributions = [[0.95, 0.05]] * 20
        truths = [0] * 20
        q = conformal_quantile(distributions, truths, 1.0, alpha=0.1)
        assert q <= 0.06  # scores all 0.05: tight bar

    def test_too_few_points_degrades_to_full_coverage(self):
        q = conformal_quantile([[0.9, 0.1]], [0], 1.0, alpha=0.05)
        assert q == 1.0  # n=1 cannot support alpha=0.05: everything conformal

    def test_singleton_dispatch_rule(self):
        included, calibrated = prediction_set([0.9, 0.08, 0.02], 1.0, 0.2)
        assert included == [0]
        ambiguous, _ = prediction_set([0.5, 0.45, 0.05], 1.0, 0.6)
        assert len(ambiguous) > 1  # ambiguous record -> fallback at runtime

    def test_report_shape(self):
        distributions = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]
        truths = [0, 1, 0]
        t = fit_temperature(distributions, truths)
        q = conformal_quantile(distributions, truths, t, alpha=0.25)
        report = calibration_report(distributions, truths, t, q, 0.25)
        assert report["n_calibration"] == 3
        assert 0.0 <= report["dispatch_rate_on_calibration"] <= 1.0
        assert report["quantile_granularity"] == 0.25


def _draw(rng, n):
    """Synthetic 3-class generative process with mixed difficulty."""
    distributions, truths = [], []
    for _ in range(n):
        true = rng.randrange(3)
        concentration = rng.choice([0.6, 0.9, 2.5, 5.0])
        logits = [rng.gauss(0, 1) for _ in range(3)]
        logits[true] += concentration
        peak = max(logits)
        exps = [math.exp(x - peak) for x in logits]
        total = sum(exps)
        distributions.append([x / total for x in exps])
        truths.append(true)
    return distributions, truths


class TestSplitCalibration:
    def test_folds_are_disjoint_and_reported(self):
        distributions = [[0.9, 0.1]] * 10
        truths = [0] * 10
        report = fit_split_calibration(distributions, truths, alpha=0.2)
        assert report["n_calibration"] == 10
        assert report["n_temperature_fold"] == 5
        assert report["n_conformal_fold"] == 5
        assert report["quantile_granularity"] == round(1 / 6, 4)
        assert "marginal" in report["guarantee"]

    def test_tiny_temperature_fold_falls_back_to_t1(self):
        distributions = [[0.8, 0.2]] * 4  # T-fold gets 2 points (< 4)
        report = fit_split_calibration(distributions, [0] * 4, alpha=0.4)
        assert report["temperature"] == 1.0

    def test_conditional_bound_reported(self):
        distributions = [[0.95, 0.05]] * 20
        report = fit_split_calibration(distributions, [0] * 20, alpha=0.2)
        rate = report["dispatch_rate_on_calibration"]
        if rate:
            assert report["conditional_error_bound_estimate"] == round(
                min(1.0, 0.2 / rate), 4
            )

    def test_empirical_marginal_coverage_holds(self):
        # Fit on 300 points, evaluate wrong-dispatch rate on 2000 fresh points
        # from the same process: must stay within alpha (+ finite-sample slack).
        rng = random.Random(7)
        cal_d, cal_y = _draw(rng, 300)
        test_d, test_y = _draw(rng, 2000)
        alpha = 0.1
        report = fit_split_calibration(cal_d, cal_y, alpha=alpha)
        t, q = report["temperature"], report["conformal_quantile"]
        wrong_dispatches = 0
        for probs, true in zip(test_d, test_y):
            included, _ = prediction_set(probs, t, q)
            if len(included) == 1 and included[0] != true:
                wrong_dispatches += 1
        assert wrong_dispatches / len(test_d) <= alpha + 0.02


class TestGuards:
    def test_bad_temperature(self):
        with pytest.raises(ValueError):
            apply_temperature([0.5, 0.5], 0.0)

    def test_bad_alpha(self):
        with pytest.raises(ValueError):
            conformal_quantile([[0.5, 0.5]], [0], 1.0, alpha=1.5)

    def test_mismatched_inputs(self):
        with pytest.raises(ValueError):
            fit_temperature([[0.5, 0.5]], [0, 1])
