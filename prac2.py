import re
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

def main():

    ## The molecules have different ranges of r so the filenames are different
    mol, r_min = selection()

    r = arange(r_min, r_min + 1.21, 0.05)
    theta = arange(70, 161, 1)
    energy = get_energies(mol, r, theta)
    
    r_eq_index, theta_eq_index, energy_eq_index = equilibrium(r, theta, energy)
    
    graph_data(r, theta, energy)

    ## Fit data along normal mode
    k_r, k_t = fit_modes(r, theta, energy, r_eq_index, theta_eq_index, energy_eq_index)

    print_frequencies(k_r, k_t, r, r_eq_index)

def selection():

    accepted = False
    while accepted == False:
        print("Please type 1 for H2O or 2 for H2S")
        inp = input()
        if inp == "1":
            accepted = True
            mol, r_min = "H2O", 0.7
        elif inp == "2":
            accepted = True
            mol, r_min = "H2S", 0.6
        else:
            print("Invalid input, try again")

    return mol, r_min

def get_energies(mol, r, theta):

    energy = []
    folder = mol + "outfiles/"

    ## Create a new filename variable each iteration of a loop to open all
    ## the necessary files 
    
    for i in range(len(r)):
        for j in range(len(theta)):
            file_name = mol + ".r"
            file_name += str(format(r[i], '.2f'))
            file_name += "theta" + str(format(theta[j], '.1f')) + ".out"

            ## Open the selected file and read its contents
            f = open(folder + file_name, "r")
            content = f.read()

            ## This regex searches for "E(RHF) =  " and finds the value after
            match = re.search(r"E\(RHF\) =\s*(-?\d+\.\d+)", content)
            energy.append(float(match.group(1)))
            
    return energy

def graph_data(r, theta, energy):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    
    Y = array(r)
    X = array(theta)
    X, Y = meshgrid(X, Y)

    Z = array(energy).reshape(Y.shape)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis
    ax.set_zlim(min(energy), max(energy))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlabel("Energy / eV")
    ax.set_xlabel("Bond angle, theta / degrees")
    ax.set_ylabel("Bond length, r / Angstrom")

    plt.show()

def equilibrium(r, theta, energy):

    energy_eq = min(energy)
    energy_eq_index = energy.index(energy_eq)
    min_energy = round(energy_eq, 4)

    theta_eq_index = energy_eq_index % (len(theta))
    theta_eq = round(theta[theta_eq_index], 1)

    r_eq_index = round(len(r) * energy_eq_index / len(energy))
    r_eq = round(r[r_eq_index], 2)


    print("***********************************")
    print()
    print("The equilibrium energy is " + str(min_energy) + " eV")
    print()
    print("Which occurs at r = " + str(r_eq) + " A, and theta = " + str(theta_eq) + " degrees")
    print()

    print("***********************************")

    return r_eq_index, theta_eq_index, energy_eq_index

def fit_modes(r, theta, energy, r_eq_index, theta_eq_index, energy_eq_index):

    ## Since E = Er(r) + Et(theta) we can fit separately the A_1 and bending modes
    ## Final values are highly sensitive to the number of points used
    ## these values give the closest for both
    distance_left, distance_right = 5, 10
    index_step = len(theta)
    r_energies, t_energies = [], []
    au_to_J = 4.35974 * pow(10, -18)
    A_to_m = pow(10, -10)
    m_to_bohr = 1/(0.529177 * pow(10,-10))
    A_to_bohr = 1/0.529177
    deg_to_rad = pi / 180

    ## Take values around the equilibrium point as a subset
    r_subset = r[r_eq_index - distance_left : r_eq_index + distance_right]
    theta_subset = theta[theta_eq_index - distance_left : theta_eq_index + distance_right]
    
    ## Changing variables to displacements from equilibrium positions
    x = (r_subset - r[r_eq_index]) * A_to_m
    t = (theta_subset - theta[theta_eq_index]) * deg_to_rad         

    ## select the corresponding values from energy list
    for i in range(-distance_left, distance_right):
        en_t = energy[energy_eq_index + i] * au_to_J
        t_energies.append(en_t)

        j = energy_eq_index + i * index_step
        en_r = energy[j] * au_to_J
        r_energies.append(en_r)

    ## Fit a quadratic potential for each series
    p_r = polyfit(x, r_energies, 2)
    p_t = polyfit(t, t_energies, 2)

    ## The polynomial fit success can be checked by this plot
    xfit = arange(min(t), max(t), 0.001 * (max(t) - min(t)))
    yfit = polyval(p_t, xfit)

    plt.scatter(t, t_energies, color="red", marker="x")
    plt.plot(xfit, yfit)
    ## plt.show()

    ## Relate the spring constant to the squared term of polynomial fit

    k_r = 2 * p_r[0]
    k_t = 2 * p_t[0]

    return k_r, k_t

def print_frequencies(k_r, k_t, r, r_eq_index):

    amu = 1.66053886 * pow(10, -27)
    hz_cm = 3.3356 * pow(10, -11)
    conv = hz_cm/(2 * pi)
    A_to_m = pow(10,-10)

    r_eq = r[r_eq_index] * A_to_m
    m_u = amu

    ## As per equations 3.1 and 3.2 in the handout, vibrational frequencies are given by
    v_r = conv * sqrt(k_r / (2 * m_u))
    v_t = conv * sqrt(k_t/ (r_eq * r_eq * 0.5 * m_u))

    ## Lit H2O: 3585, 1885 H2S: 2615, 1183
    print()
    print("Vibrational Frequencies are: ")
    print("Symmetric stretch: " + str(round(v_r, 1)) + " cm^-1")
    print("Bending Mode: " + str(round(v_t, 1)) + " cm^-1")
    print()

main()