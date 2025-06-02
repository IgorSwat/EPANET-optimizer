defmodule Zakharov do
  import Nx.Defn
  use SharedMacro

  defn get_d, do: @d
  def optimal_point(d), do: Nx.broadcast(0, {1, d})
  defn evaluate_nx_matrix_defn(u) do
    indices = Nx.iota({Nx.axis_size(u, 1)}) |> Nx.add(1) |> Nx.multiply(0.5)
    sum_square_term = Nx.sum(u ** 2, axes: [1])
    linear_term = Nx.sum(indices * u, axes: [1])
    quadratic_term = linear_term ** 2
    quartic_term = linear_term ** 4
    sum_square_term + quadratic_term + quartic_term
  end
end
