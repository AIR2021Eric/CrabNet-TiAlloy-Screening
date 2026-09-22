import numpy as np

# Elemental reference constants (GPa, g/cm3) - same source as Diagnostic 2
EL = {
    "Ti": dict(K=110, G=42, rho=4.51), "Al": dict(K=76, G=26, rho=2.70),
    "V":  dict(K=160, G=47, rho=6.00), "Mo": dict(K=230, G=126, rho=10.22),
    "Zr": dict(K=91,  G=33, rho=6.52), "Cr": dict(K=160, G=115, rho=7.19),
    "Fe": dict(K=170, G=82, rho=7.87),
}

def rom_vrh(comp: dict):
    """comp = atomic fractions, e.g. {'Ti':0.90,'Al':0.06,'V':0.04}"""
    f = {el: n/sum(comp.values()) for el, n in comp.items()}
    Kv = sum(f[e]*EL[e]["K"] for e in f); Kr = 1/sum(f[e]/EL[e]["K"] for e in f)
    Gv = sum(f[e]*EL[e]["G"] for e in f); Gr = 1/sum(f[e]/EL[e]["G"] for e in f)
    K, G = (Kv+Kr)/2, (Gv+Gr)/2
    E = 9*K*G/(3*K+G)
    rho = 1/sum(f[e]/EL[e]["rho"] for e in f)
    return K, G, E, rho

for name, comp in [("Ti-6Al-4V", {"Ti":90,"Al":6,"V":4}),
                   ("Ti-5553",   {"Ti":79,"Al":12,"V":6,"Mo":2,"Cr":2})]:  # at% approx
    K, G, E, rho = rom_vrh(comp)
    print(f"{name}: ROM K={K:.0f} G={G:.0f} E={E:.0f} rho={rho:.2f}")