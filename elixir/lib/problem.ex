defmodule Problem do
  @derive {Nx.Container, containers: [:l, :u, :shift]}
  @enforce_keys [:d]
  defstruct name: nil,
            d: 10,
            fun: nil,
            shift: nil,
            l: nil,
            u: nil

  @type t :: %__MODULE__{
          name: String.t() | nil,
          d: non_neg_integer(),
          fun: (Nx.Tensor.t() -> Nx.Tensor.t()) | nil,
          shift: Nx.Tensor.t() | nil,
          l: Nx.Tensor.t() | nil,
          u: Nx.Tensor.t() | nil
        }

  use SharedMacro
  def n, do: @n

  @doc """
  Creates a new problem instance with a random shift and a wrapped evaluation function.

  Defaults:
    - `l`: A d-dimensional tensor of -10, unless provided in opts.
    - `u`: A d-dimensional tensor of 10, unless provided in opts.
    - `shift`: A random d-dimensional tensor with values uniformly drawn from [-80, 80]
      using the `Nx.Random.uniform/4` function.

  The provided original function (`fun`) is wrapped using `Nx.Defn.jit/1` so that the input is shifted
  before evaluation. The original `fun` is expected to map an input tensor of shape {n, d} to an output tensor of shape {n}.
  """
  @spec new(map()) :: t()
  def new(%{d: d, fun: orig_fun} = opts) when is_function(orig_fun, 1) do
    # Generate a random shift tensor of shape {d}.
    {shift_tensor, _new_key} =
      Nx.Random.uniform(Nx.Random.key(12), -0.8, 0.8, shape: {d})

    default_fields = %{
      l: Nx.broadcast(-10, {d}),
      u: Nx.broadcast(10, {d}),
      shift: shift_tensor
    }

    merged = Map.merge(default_fields, opts)

    wrapped_fun =
      Nx.Defn.jit(fn x, shift ->
        # x: {n, d}
        # Transform shift (shape {d}) into shape {1, d}, then subtract:
        bshift = Nx.new_axis(shift, 0)
        x_shifted = Nx.subtract(x, bshift)
        orig_fun.(x_shifted)
      end)

    struct(__MODULE__, Map.put(merged, :fun, wrapped_fun))
  end
end
