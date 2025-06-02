defmodule BentCigar do
  import Nx.Defn
  use SharedMacro

  # Returns the dimension provided by a module attribute @d,
  # assuming it is set via SharedMacro.
  defn get_d, do: @d
  def optimal_point(d), do: Nx.broadcast(0, {1, d})
  defn evaluate_nx_matrix_defn(u) do
    # Assume u is of shape {batch, dimensions}
    batch_size = Nx.axis_size(u, 0)
    d = Nx.axis_size(u, 1)

    # Extract the first coordinate: x₁ and compute its square.
    x1 = Nx.slice(u, [0, 0], [batch_size, 1])
    first_term = x1 ** 2

    # Extract the remaining coordinates: x₂, x₃, ..., x_D,
    # compute the sum of their squares, then scale it by 10⁶.
    rest = Nx.slice(u, [0, 1], [batch_size, d - 1])
    rest_term = Nx.sum(rest ** 2, axes: [1]) * 1_000_000

    # Reshape the first term from {batch, 1} to {batch} and add the terms.
    first_term_flat = Nx.reshape(first_term, {batch_size})
    first_term_flat + rest_term
  end
end
