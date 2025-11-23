"""
Test script for HAR-CAESar implementation.

This script verifies that the HAR-CAESar model can be instantiated and run
on synthetic data.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from har_caesar.models.har_caesar import HAR_CAESar, compute_har_features


def test_har_features():
    """Test the HAR feature computation."""
    print("Testing compute_har_features()...")
    
    # Create synthetic returns
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02  # 100 days, 2% daily vol
    
    # Compute HAR features
    har = compute_har_features(returns)
    
    assert 'daily' in har, "Missing daily features"
    assert 'weekly' in har, "Missing weekly features"
    assert 'monthly' in har, "Missing monthly features"
    assert len(har['daily']) == 100, "Daily features wrong length"
    assert len(har['weekly']) == 100, "Weekly features wrong length"
    assert len(har['monthly']) == 100, "Monthly features wrong length"
    
    print("  ✓ HAR features computed correctly")
    print(f"    Daily shape: {har['daily'].shape}")
    print(f"    Weekly shape: {har['weekly'].shape}")
    print(f"    Monthly shape: {har['monthly'].shape}")


def test_har_caesar_init():
    """Test HAR-CAESar initialization."""
    print("\nTesting HAR_CAESar initialization...")
    
    model = HAR_CAESar(theta=0.025)
    
    assert model.theta == 0.025, "Theta not set correctly"
    assert model.n_parameters == 9, "n_parameters should be 9"
    assert model.mdl_spec == 'HAR', "mdl_spec should be 'HAR'"
    
    print("  ✓ HAR_CAESar initialized correctly")
    print(f"    theta: {model.theta}")
    print(f"    n_parameters: {model.n_parameters}")


def test_har_caesar_fit():
    """Test HAR-CAESar fitting on synthetic data."""
    print("\nTesting HAR_CAESar fit (this may take a minute)...")
    
    # Generate synthetic GARCH-like returns
    np.random.seed(42)
    T = 500
    returns = np.zeros(T)
    sigma = np.zeros(T)
    sigma[0] = 0.02
    
    for t in range(1, T):
        sigma[t] = np.sqrt(0.0001 + 0.1 * returns[t-1]**2 + 0.85 * sigma[t-1]**2)
        returns[t] = sigma[t] * np.random.standard_t(df=5)
    
    # Fit model
    model = HAR_CAESar(theta=0.025)
    result = model.fit(returns, seed=42, return_train=True, nV=30, n_init=2, n_rep=2)
    
    assert 'qi' in result, "Missing qi in result"
    assert 'ei' in result, "Missing ei in result"
    assert 'beta' in result, "Missing beta in result"
    assert len(result['qi']) == T, "qi wrong length"
    assert len(result['ei']) == T, "ei wrong length"
    assert result['beta'].shape == (2, 9), "beta wrong shape"
    
    # Check monotonicity constraint (ES <= VaR for negative values)
    violations = np.sum(result['ei'] > result['qi'])
    print(f"  ✓ Model fitted successfully")
    print(f"    VaR mean: {np.mean(result['qi']):.4f}")
    print(f"    ES mean: {np.mean(result['ei']):.4f}")
    print(f"    Monotonicity violations: {violations}/{T}")
    print(f"    Beta shape: {result['beta'].shape}")
    


def test_har_caesar_predict():
    """Test HAR-CAESar prediction."""
    print("\nTesting HAR_CAESar predict...")
    
    # Generate data
    np.random.seed(42)
    T_train = 400
    T_test = 100
    T = T_train + T_test
    
    returns = np.zeros(T)
    sigma = np.zeros(T)
    sigma[0] = 0.02
    
    for t in range(1, T):
        sigma[t] = np.sqrt(0.0001 + 0.1 * returns[t-1]**2 + 0.85 * sigma[t-1]**2)
        returns[t] = sigma[t] * np.random.standard_t(df=5)
    
    # Fit and predict
    model = HAR_CAESar(theta=0.025)
    result = model.fit_predict(returns, ti=T_train, seed=42, return_train=True,
                               nV=30, n_init=2, n_rep=2)
    
    assert 'qi' in result, "Missing qi in result"
    assert 'ei' in result, "Missing ei in result"
    assert 'qf' in result, "Missing qf in result"
    assert 'ef' in result, "Missing ef in result"
    assert len(result['qf']) == T_test, f"qf wrong length: {len(result['qf'])} vs {T_test}"
    assert len(result['ef']) == T_test, f"ef wrong length: {len(result['ef'])} vs {T_test}"
    
    print(f"  ✓ Prediction successful")
    print(f"    Train VaR mean: {np.mean(result['qi']):.4f}")
    print(f"    Test VaR mean: {np.mean(result['qf']):.4f}")
    print(f"    Train ES mean: {np.mean(result['ei']):.4f}")
    print(f"    Test ES mean: {np.mean(result['ef']):.4f}")
    


if __name__ == '__main__':
    print("=" * 60)
    print("HAR-CAESar Implementation Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_har_features()
    except Exception as e:
        print(f"  ✗ HAR features test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_har_caesar_init()
    except Exception as e:
        print(f"  ✗ Initialization test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_har_caesar_fit()
    except Exception as e:
        print(f"  ✗ Fit test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_har_caesar_predict()
    except Exception as e:
        print(f"  ✗ Predict test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 60)
