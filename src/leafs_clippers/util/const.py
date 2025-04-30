C_LIGHT = 299792.458  # c in km/s
H_ERG = 6.62607015e-27  # Plancks constant (erg.s)
C_AA = 299792458 * 1.0e10  # in AA/s
M_SOL = 1.989e33  # solar mass in g
R_SOL = 6.955e10  # solar radius in cm
G_GRAV = 6.67259e-8  # gravitational constant in cm^3 g^-1 s^-2

KEY_TO_LABEL_DICT = {
    "density": r"$\rho$ (g cm$^{-3}$)",
    "temp": r"$T$ (K)",
    "pressure": r"$P$ (dyn cm$^{-2}$)",
    "velx": r"$v_x$ (cm s$^{-1}$)",
    "vely": r"$v_y$ (cm s$^{-1}$)",
    "velz": r"$v_z$ (cm s$^{-1}$)",
    "ye": r"$Y_e$",
    "xnuc01": r"$xnuc01$",
    "xnuc02": r"$xnuc02$",
    "xnuc03": r"$xnuc03$",
    "xnuc04": r"$xnuc04$",
    "xnuc05": r"$xnuc05$",
    "xnuc06": r"$xnuc06$",
    "energy": r"$\epsilon$ (erg g$^{-1}$)",
    "Amean": r"$A_{\rm mean}$",
    "q_sgs": r"$q_{\rm sgs}$ (cm s$^{-1}$)",
}

KEY_TO_CMAP_DICT = {
    "density": "inferno",
    "temp": "afmhot",
    "pressure": "viridis",
    "velx": "coolwarm",
    "vely": "coolwarm",
    "velz": "coolwarm",
    "ye": "cividis",
    "xnuc01": "ocean",
    "xnuc02": "ocean",
    "xnuc03": "ocean",
    "xnuc04": "ocean",
    "xnuc05": "ocean",
    "xnuc06": "ocean",
    "energy": "magma",
    "mach": "BuPu_r",
}


def FLOAT_FMT(x, pos):
    return "{:.2f}".format(x)


def INT_FMT(x, pos):
    return "{:d}".format(int(x))


KEY_TO_FMT_DICT = {
    "density": FLOAT_FMT,
    "temp": FLOAT_FMT,
    "pressure": FLOAT_FMT,
    "velx": FLOAT_FMT,
    "vely": FLOAT_FMT,
    "velz": FLOAT_FMT,
    "ye": FLOAT_FMT,
    "xnuc01": FLOAT_FMT,
    "xnuc02": FLOAT_FMT,
    "xnuc03": FLOAT_FMT,
    "xnuc04": FLOAT_FMT,
    "xnuc05": FLOAT_FMT,
    "xnuc06": FLOAT_FMT,
    "Amean": INT_FMT,
}
