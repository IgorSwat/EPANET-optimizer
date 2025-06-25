import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_convergence(file_path, output_path="./plots/wso_perturbed_convergence.png"):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Plik '{file_path}' nie istnieje.")

    # Wczytanie danych fitness
    with open(file_path, 'r') as f:
        fitness_values = [float(line.strip()) for line in f if line.strip()]

    iterations = np.arange(1, len(fitness_values) + 1)

    # Tworzenie wykresu
    plt.figure(figsize=(8, 5))
    plt.xscale(value='log')
    plt.plot(iterations, fitness_values, marker='o', linestyle='-')
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu (fitness)")
    plt.title("Zbieżność algorytmu WSO (96 rekinów, 10000 iteracji)")
    plt.grid(True)
    plt.tight_layout()

    # Zapis wykresu
    plt.savefig(output_path, dpi=300)
    print(f"Wykres zapisany do: {output_path}")

# Użycie: podaj ścieżkę do pliku jako argument lub edytuj poniżej
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python plot_wso.py <ścieżka_do_pliku_z_fitness>")
    else:
        input_path = sys.argv[1]
        plot_convergence(input_path)