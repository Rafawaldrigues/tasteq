import os

from hypothesis import HealthCheck, settings

# Rendering audio is slow compared to typical hypothesis targets, so examples
# are budgeted per profile and the deadline is disabled.
settings.register_profile(
    "dev", max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "ci", max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "thorough", max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
