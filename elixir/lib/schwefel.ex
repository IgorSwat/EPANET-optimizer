defmodule Schwefel do
  import Nx.Defn
  @a 418.9829
  # SharedMacro should define @d if you want to fix the number of dimensions.
  use SharedMacro

  defn get_d, do: @d

  defn evaluate_nx_matrix_defn(u) do
    # For each value, compute x_i * sin(sqrt(abs(x_i)))
    term = Nx.multiply(u, Nx.sin(Nx.sqrt(Nx.abs(u))))

    # Multiply the constant by the number of dimensions and subtract the sum over the 2nd axis.
    Nx.subtract(@a * get_d(), Nx.sum(term, axes: [1]))
  end
end
