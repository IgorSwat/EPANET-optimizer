import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
INPUT_FILE = 'metrics.txt'  # Zaktualizuj, jeśli inna ścieżka
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---

def load_results(filepath):
    """
    Wczytuje wyniki WSO z pliku tekstowego.
    Oczekuje linii: tag max_iterations num_sharks dimension problem time_ms fitness
    Pomija puste i niepasujące linie.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Sprawdź, czy mamy dokładnie 7 elementów
            if len(parts) != 7:
                continue
            tag, max_iter, sharks, dim, problem, time_ms, fitness = parts
            try:
                records.append({
                    'tag': tag,
                    'max_iterations': int(max_iter),
                    'num_sharks': int(sharks),
                    'dimension': int(dim),
                    'problem': problem,
                    'time_ms': float(time_ms),
                    'fitness': float(fitness)
                })
            except ValueError:
                # Pomijamy linie z błędnymi danymi
                continue
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"Brak poprawnie wczytanych rekordów z pliku '{filepath}'")
    return df

# --- Statistical Analysis ---

def compute_statistics(df):
    """
    Oblicza statystyki opisowe pogrupowane po problemie.
    Zwraca DataFrame ze średnimi, odch. std, min, max.
    """
    stats = df.groupby('problem').agg(
        runs=('fitness', 'count'),
        mean_time_ms=('time_ms', 'mean'),
        std_time_ms=('time_ms', 'std'),
        mean_fitness=('fitness', 'mean'),
        std_fitness=('fitness', 'std'),
        min_fitness=('fitness', 'min'),
        max_fitness=('fitness', 'max')
    ).reset_index()
    return stats

# --- Plotting functions ---

def plot_bar(stats, x, y, ylabel, title, filename):
    fig, ax = plt.subplots()
    ax.bar(stats[x], stats[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)


def plot_box(df, column, by, title, filename):
    fig, ax = plt.subplots()
    df.boxplot(column=column, by=by, ax=ax, rot=45)
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel(column)
    plt.suptitle('')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)


def plot_scatter(df, x, y, title, filename):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)

# --- Main ---

def main():
    try:
        df = load_results(INPUT_FILE)
    except Exception as e:
        print(e)
        return

    print(f"Wczytano {len(df)} rekordów dla {df['problem'].nunique()} problemów.")

    stats = compute_statistics(df)
    print("Statystyki opisowe wg problemu:")
    print(stats)
    stats.to_csv('wso_summary_statistics.csv', index=False)

    # Tworzenie wykresów
    plot_bar(stats, x='problem', y='mean_fitness',
             ylabel='Średnia wartość funkcji celu',
             title='Mean Fitness per Problem',
             filename='mean_fitness_per_problem.png')

    plot_bar(stats, x='problem', y='mean_time_ms',
             ylabel='Średni czas [ms]',
             title='Mean Computation Time per Problem',
             filename='mean_time_per_problem.png')

    plot_box(df, column='fitness', by='problem',
             title='Rozkład Fitness wg Problem',
             filename='fitness_boxplot_by_problem.png')

    plot_box(df, column='time_ms', by='problem',
             title='Rozkład Czasu wg Problem',
             filename='time_boxplot_by_problem.png')

    plot_scatter(df, x='time_ms', y='fitness',
                 title='Czas vs Fitness',
                 filename='time_vs_fitness_scatter.png')

    print(f"Wszystkie wykresy zapisane w katalogu '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()