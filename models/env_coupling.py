"""Environmental-disturbance -> process-operating-point adapter (L0)."""

class DisturbanceInputs:
    def __init__(self, T_ambient_C=0.0, h_conv=0.0, rain_cooling=0.0,
                 ingress_dilution=0.0, enabled=False):
        self.T_ambient_C = T_ambient_C
        self.h_conv = h_conv
        self.rain_cooling = rain_cooling
        self.ingress_dilution = ingress_dilution
        self.enabled = enabled

def disturbance_from_environment(env_state, crate_state):
    # deterministic pure mapping; zero when no env / disabled
    return DisturbanceInputs(enabled=False)
