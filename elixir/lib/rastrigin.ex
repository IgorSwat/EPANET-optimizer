defmodule Rastrigin do
  import Nx.Defn
  @a 10
  use SharedMacro
  defn get_d, do: @d
  def optimal_point(d), do: Nx.broadcast(0, {1, d})
  defn evaluate_nx_matrix_defn(u) do
    # Compute the Rastrigin function for each row:
    # f(x) = A * d + sum(x_i^2 - A * cos(2 * pi * x_i)) across all features.
    constant_term = @a * get_d()
    sum_term = Nx.sum(u ** 2 - @a * Nx.cos(2 * @pi * u), axes: [1])
    constant_term + sum_term
  end
end
