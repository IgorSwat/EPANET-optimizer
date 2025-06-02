defmodule Rosenbrock do
  import Nx.Defn
  @a 1
  @b 100
  use SharedMacro
  defn get_d, do: @d
  def optimal_point(d), do: Nx.broadcast(1, {1, d})
  defn evaluate_nx_matrix_defn(u) do

    x1 = Nx.slice(u, [0, 0], [Nx.axis_size(u, 0), get_d() - 1])
    x2 = Nx.slice(u, [0, 1], [Nx.axis_size(u, 0), get_d() - 1])

    Nx.sum((@a - x1) ** 2 + @b * (x2 - x1 ** 2) ** 2, axes: [1])
  end

end
