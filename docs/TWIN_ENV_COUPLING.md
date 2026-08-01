# Env-coupling mapping (L0)
DisturbanceInputs from env/crate observations; applied to thermal balance.
Equations: T_ambient, h_conv(wind), rain_cooling, ingress_dilution.
Defaults: coupling off => zero disturbance, byte-identical EKF.
Fail-safe: operating twin storm-mode still binds (ShutdownRequest + current=0).
