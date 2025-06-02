defmodule Levy do
  import Nx.Defn
  use SharedMacro

  defn get_d, do: @d
  def optimal_point(d), do: Nx.broadcast(1, {1, d})
  defn evaluate_nx_matrix_defn(u) do
      # u is an {n, d} matrix.
      # Transform u into w via the usual Levy transformation.
      w = 1 + (u - 1) / 4
      d = get_d()

      # First term: sin²(π * w₁)
      first_component = Nx.slice_along_axis(w, 0, 1, axis: 1)
      first_term =
        first_component
        |> Nx.multiply(@pi)
        |> Nx.sin()
        |> Nx.pow(2)
        |> Nx.squeeze(axes: [1])

      # Last term: (w_d - 1)² [1 + sin²(2π * w_d)]
      last_component = Nx.slice_along_axis(w, d - 1, 1, axis: 1)
      last_term =
        last_component
        |> Nx.subtract(1)
        |> Nx.pow(2)
        |> Nx.multiply(1 + (last_component |> Nx.multiply(2 * @pi) |> Nx.sin() |> Nx.pow(2)))
        |> Nx.squeeze(axes: [1])

      # Middle terms: Σ (wᵢ - 1)² [1 + 10 sin²(π * wᵢ + 1)]
      middle_slice = Nx.slice_along_axis(w, 0, d - 1, axis: 1)
      middle_terms =
        middle_slice
        |> Nx.subtract(1)
        |> Nx.pow(2)
        |> Nx.multiply(1 + 10 * (middle_slice |> Nx.multiply(@pi) |> Nx.add(1) |> Nx.sin() |> Nx.pow(2)))
        |> Nx.sum(axes: [1])

      first_term + middle_terms + last_term
    end

  defn evaluate_nx_defn(u) do
    w = 1 + (u - 1) / 4
    d = get_d()

    # First term: sin²(π w₁)
    first_component = Nx.slice_along_axis(w, 0, 1, axis: 1)
    first_term = Nx.sin(@pi * first_component) |> Nx.pow(2)

    # Last term: (w_D - 1)² [1 + sin²(2 π w_D)]
    last_component = Nx.slice_along_axis(w, d - 1, 1, axis: 1)
    last_term =
      (last_component - 1)
      |> Nx.pow(2)
      |> Nx.multiply(1 + (Nx.sin(2 * @pi * last_component) |> Nx.pow(2)))

    # Middle terms: Σ (w_i - 1)² [1 + 10 sin²(π w_i + 1)]
    middle_slice = Nx.slice_along_axis(w, 0, d - 1, axis: 1)
    middle_terms =
      (middle_slice - 1)
      |> Nx.pow(2)
      |> Nx.multiply(
            1 +
              10 *
                (Nx.sin(@pi * middle_slice + 1)
                |> Nx.pow(2))
          )
      |> Nx.sum(axes: [1])

    first_term + middle_terms + last_term
  end
end
