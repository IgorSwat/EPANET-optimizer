defmodule Rosenbrock do
  import Nx.Defn

  def evaluate_nx(u \\ Nx.tensor([1.0, 1.0, 1.0], type: {:f, 32}), a \\ 1.0, b \\ 100.0) do
    d = Nx.size(u)
    true_solution = Nx.tensor(List.duplicate(2.0, d))
    evaluate_nx_defn(u, true_solution, a, b)
  end

  defn evaluate_nx_defn(u, true_solution, a, b) do
    adjusted_u = Nx.subtract(u, true_solution)
    size_u = Nx.size(adjusted_u)
    
    # Note: Nx.slice/4 expects start indices and slice lengths as lists.
    x1 = Nx.slice(adjusted_u, [0], [size_u - 1])
    x2 = Nx.slice(adjusted_u, [1], [size_u - 1])
    
    Nx.sum((a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2)
  end
end
