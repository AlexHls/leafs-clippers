from io import StringIO

import numyp as np

from leafs_clippers.util import utilities as util


class AYANNWeakRate:
    def __init__(self, rate):
        self.input = rate.input
        self.output = rate.output
        self.q1 = rate.q1
        self.q2 = rate.q2
        self.reverse = rate.reverse
        self.weak = rate.weak
        self.temp = rate.temp
        self.rhoye = rate.rhoye
        self.lambda1 = rate.lambda1
        self.lambda2 = rate.lambda2
        self.lambda3 = rate.lambda3
        self.temp_len = rate.temp_len
        assert self.temp_len == len(
            self.temp
        ), "Number of temperatures is not equal to temp_len ({} {}, {}, {})".format(
            self.temp_len, len(self.temp), self.input, self.output
        )
        assert (
            len(self.lambda1) == 143
            and len(self.lambda2) == 143
            and len(self.lambda3) == 143
        ), "({:s}, {:s}) Number of lambda1 rates is not 143, but {:d}".format(
            self.input, self.output, len(self.lambda1)
        )
        assert (
            len(self.rhoye) * self.temp_len == 143
        ), "Number of rates is not 143 ({} {}, {}, {})".format(
            len(self.rhoye), len(self.temp), self.input, self.output
        )

    def get_input_output(self):
        return self.input, self.output


class AYANNWeakRates:
    def __init__(self, rates):
        self.rates = {}
        for rate in rates:
            self.rates[rate.get_input_output()] = AYANNWeakRate(rate)

    def print_rates(self):
        for rate in list(self.rates.keys()):
            print(rate)
        return

    def get_rates(self):
        return [self.rates[r] for r in list(self.rates.keys())]

    def write_rates(self, filename):
        with open(filename, "w") as f:
            for r in self.rates:
                rate = self.rates[r]
                # Rate header
                f.write(rate.input.rjust(5))
                f.write(rate.output.rjust(5))
                f.write("{:.5f}".format(rate.q1).rjust(12))
                f.write("{:.5f}".format(rate.q2).rjust(12))
                f.write("  ")
                f.write(str(rate.reverse).rjust(2))
                f.write("\n")

                if len(rate.rhoye) * rate.temp_len != 143:
                    print(rate.input, rate.output, len(rate.rhoye), rate.temp_len)
                assert (
                    len(rate.rhoye) * rate.temp_len == 143
                ), "Number of rates is not 143"

                # Rate data
                for i, rho in enumerate(rate.rhoye):
                    for j, temp in enumerate(rate.temp):
                        ind = i * rate.temp_len + j
                        f.write("{:.2f}".format(temp).rjust(6))
                        f.write("{:.2f}".format(rho).rjust(6))
                        f.write("{:.3f}".format(rate.lambda1[ind]).rjust(9))
                        f.write("{:.3f}".format(rate.lambda2[ind]).rjust(9))
                        f.write("{:.3f}".format(rate.lambda3[ind]).rjust(9))
                        f.write("\n")
        return


class NKKWeakRate(AYANNWeakRate):
    def __init__(self, lines):
        line = lines[0]
        number1 = line[:37].strip()
        number2 = line[37:].strip()
        line = lines[1]
        name1 = line[:40].strip()
        name2 = line[52:].strip()
        self.input = name1.lower() + number1
        self.output = name2.lower() + number2
        self.q1 = 0.0  # Dummy value
        self.q2 = 0.0  # Dummy value
        self.reverse = -1  # Dummy value
        self.weak = True
        rhoyecol, tempcol, self.lambda1, self.lambda2, self.lambda3 = np.loadtxt(
            StringIO("".join(lines[8:])), usecols=(0, 1, 3, 4, 5), unpack=True
        )

        self.temp = tempcol[:12:]
        self.rhoye = rhoyecol[::12]
        self.temp_len = 12

    def extrapolate(self, temp=100):
        # Constant extrapolation
        self.temp = np.append(self.temp, temp)
        self.temp_len = 13
        self.lambda1 = util.extrapolate_nkk_aux(self.lambda1)
        self.lambda2 = util.extrapolate_nkk_aux(self.lambda2)
        self.lambda3 = util.extrapolate_nkk_aux(self.lambda3)


class NKKWeakRates(AYANNWeakRates):
    def __init__(self, filename):
        self.rates = {}

        with open(filename) as f:
            lines = f.readlines()

        num_rates = len(lines) / 151

        # Check if num_rates is an integer
        if num_rates != int(num_rates):
            raise ValueError("Number of lines in file is not a multiple of 151")

        for i in range(int(num_rates)):
            rate_lines = lines[i * 151 : (i + 1) * 151]
            rate = NKKWeakRate(rate_lines)
            self.rates[rate.get_input_output()] = rate

    def extrapolate(self, temp=100):
        for rate in list(self.rates.keys()):
            self.rates[rate].extrapolate(temp)

    def print_rates(self):
        for rate in list(self.rates.keys()):
            print(rate)
        return

    def write_rates(self, filename):
        with open(filename, "w") as f:
            for r in self.rates:
                rate = self.rates[r]
                # Rate header
                f.write(rate.input.rjust(5))
                f.write(rate.output.rjust(5))
                f.write("{:.5f}".format(rate.q1).rjust(12))
                f.write("{:.5f}".format(rate.q2).rjust(12))
                f.write("  ")
                f.write(str(rate.reverse).rjust(2))
                f.write("\n")

                # Rate data
                for i, rho in enumerate(rate.rhoye):
                    for j, temp in enumerate(rate.temp):
                        ind = i * rate.temp_len + j
                        f.write("{:.2f}".format(temp).rjust(6))
                        f.write("{:.2f}".format(rho).rjust(6))
                        f.write("{:.3f}".format(rate.lambda1[ind]).rjust(9))
                        f.write("{:.3f}".format(rate.lambda2[ind]).rjust(9))
                        f.write("{:.3f}".format(rate.lambda3[ind]).rjust(9))
                        f.write("\n")
        return


class ODAWeakRate(AYANNWeakRate):
    def __init__(self, lines, species, reverse):
        assert reverse in [-1, 1], "Reverse must be -1 or 1"
        line = lines[0]
        z = line[23:25].strip()
        a = line[28:30].strip()
        name = species[(species["Z"] == int(z)) & (species["A"] == int(a))]["Name"]
        if len(name) == 0:
            raise ValueError("Species not found in species file")

        z_input = int(z) + 1
        name_input = species[(species["Z"] == z_input) & (species["A"] == int(a))][
            "Name"
        ]

        if reverse == 1:
            self.input = name.values[0].lower()
            self.output = name_input.values[0].lower()
        else:
            self.input = name_input.values[0].lower()
            self.output = name.values[0].lower()

        self.q1 = float(line[33:].strip()) * -1 * reverse
        self.q2 = 0.0  # Dummy value
        self.reverse = reverse
        self.weak = True
        if reverse == 1:
            cols = (0, 1, 6, 7, 8)
        else:
            cols = (0, 1, 3, 4, 5)
        tempcol, rhoyecol, self.lambda2, self.lambda1, self.lambda3 = np.loadtxt(
            StringIO("".join(lines[2:62] + lines[63:])), usecols=cols, unpack=True
        )

        self.temp = tempcol[:12:]
        self.rhoye = rhoyecol[::12]
        self.temp_len = 12

    def extrapolate(self, temp=100):
        # Constant extrapolation
        self.temp = np.append(self.temp, temp)
        self.temp_len = 13
        self.lambda1 = util.extrapolate_oda_aux(self.lambda1)
        self.lambda2 = util.extrapolate_oda_aux(self.lambda2)
        self.lambda3 = util.extrapolate_oda_aux(self.lambda3)


class ODAWeakRates(AYANNWeakRates):
    def __init__(self, filename, species_file, load_reverse=True):
        self.rates = {}

        with open(filename) as f:
            lines = f.readlines()

        num_rates = len(lines) / 135

        # Check if num_rates is an integer
        if num_rates != int(num_rates):
            raise ValueError("Number of lines in file is not a multiple of 135")

        species = util.load_species(species_file)
        species["N"] = species["A"] - species["Z"]

        for i in range(int(num_rates)):
            rate_lines = lines[i * 135 : (i + 1) * 135]
            rate = ODAWeakRate(rate_lines, species, 1)
            self.rates[rate.get_input_output()] = rate
            # Reverse rate
            if load_reverse:
                rate = ODAWeakRate(rate_lines, species, -1)
                self.rates[rate.get_input_output()] = rate

    def extrapolate(self, temp=100):
        for rate in list(self.rates.keys()):
            self.rates[rate].extrapolate(temp)

    def print_rates(self):
        for rate in list(self.rates.keys()):
            print(rate)
        return

    def write_rates(self, filename):
        with open(filename, "w") as f:
            for r in self.rates:
                rate = self.rates[r]
                # Rate header
                f.write(rate.input.rjust(5))
                f.write(rate.output.rjust(5))
                f.write("{:.5f}".format(rate.q1).rjust(12))
                f.write("{:.5f}".format(rate.q2).rjust(12))
                f.write("  ")
                f.write(str(rate.reverse).rjust(2))
                f.write("\n")

                # Rate data
                for i, rho in enumerate(rate.rhoye):
                    for j, temp in enumerate(rate.temp):
                        ind = i * rate.temp_len + j
                        f.write("{:.2f}".format(temp).rjust(6))
                        f.write("{:.2f}".format(rho).rjust(6))
                        f.write("{:.3f}".format(rate.lambda1[ind]).rjust(9))
                        f.write("{:.3f}".format(rate.lambda2[ind]).rjust(9))
                        f.write("{:.3f}".format(rate.lambda3[ind]).rjust(9))
                        f.write("\n")
        return


class LMPWeakRate(AYANNWeakRate):
    def __init__(self, lines):
        line = lines[0]
        name = line[:5].strip()
        self.input = name
        name = line[5:10].strip()
        self.output = name
        self.q1 = float(line[12:24])
        self.q2 = float(line[24:36])
        self.reverse = int(line[36:])
        self.weak = True
        tempcol, rhoyecol, self.lambda1, self.lambda2, self.lambda3 = np.loadtxt(
            StringIO("".join(lines[1:])), usecols=(0, 1, 2, 3, 4), unpack=True
        )

        self.temp = tempcol[:13:]
        self.rhoye = rhoyecol[::13]
        self.temp_len = 13


class LMPWeakRates(AYANNWeakRates):
    def __init__(self, filename):
        self.rates = {}

        with open(filename) as f:
            lines = f.readlines()

        num_rates = len(lines) / 144

        # Check if num_rates is an integer
        if num_rates != int(num_rates):
            raise ValueError("Number of lines in file is not a multiple of 144")

        for i in range(int(num_rates)):
            rate_lines = lines[i * 144 : (i + 1) * 144]
            rate = LMPWeakRate(rate_lines)
            self.rates[rate.get_input_output()] = rate
