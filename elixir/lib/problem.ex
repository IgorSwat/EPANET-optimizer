defmodule Problem do
  @derive {Nx.Container, containers: [:l, :u, :shift, :optimum]}
  @enforce_keys [:d]
  defstruct name: nil,
            d: 10,
            fun: nil,
            shift: nil,
            l: nil,
            u: nil,
            # The unshifted optimum of the original function; default is ones, but users can supply any.
            optimum: nil

  @type t :: %__MODULE__{
          name: String.t() | nil,
          d: non_neg_integer(),
          fun: (Nx.Tensor.t() -> Nx.Tensor.t()) | nil,
          shift: Nx.Tensor.t() | nil,
          l: Nx.Tensor.t() | nil,
          u: Nx.Tensor.t() | nil,
          optimum: Nx.Tensor.t() | nil
        }

  use SharedMacro
  def n, do: @n

  @doc """
  Creates a new problem instance where the overall optimum (i.e. shift + optimum) is always within [-80, 80]^d.

  **Defaults:**

    - `l`: A d-dimensional tensor of -80.
    - `u`: A d-dimensional tensor of 80.
    - `optimum`: The original (unshifted) optimum.
      Defaults to a d-dimensional tensor of ones but can be provided by the caller (for example, it might be `420^d`).

  **Mechanism:**

  1. First, a target optimum point is randomly chosen from the uniform distribution on `[-80, 80]^d`.
  2. Then, the module computes the shift as:

     ```
     shift = target_optimum - optimum
     ```

     so that when the wrapped function evaluates an input \(x\) as `orig_fun.(x - shift)`, its optimum becomes:

     ```
     optimal_point = shift + optimum = target_optimum
     ```

     guaranteeing that the optimum is always in \([-80,80]^d\).

  3. The wrapped function is then defined using `Nx.Defn.jit/1` so that an input tensor `x` is first shifted (by subtracting `shift`) before being passed to the original function.

  **Usage Note:**

  If you supply an `optimum` that is not in \([-80,80]^d\) (for example, `420^d`), the random target optimum (sampled from [-80,80]^d) will still force the overall optimum,
  computed as `shift + optimum`, to lie within the desired range.
  """
  @spec new(map()) :: t()
  def new(%{d: d, fun: orig_fun} = opts) when is_function(orig_fun, 1) do
    # Default bounds for the overall search space, now matching the [-80,80]^d criterion.
    default_fields = %{
      l: Nx.broadcast(-100, {d}),
      u: Nx.broadcast(100, {d}),
      # Default original optimum is a ones vector (can be replaced by caller)
      optimum: Nx.broadcast(1, {d})
    }

    merged = Map.merge(default_fields, opts)
    unshifted_optimum = merged.optimum

    # Generate the target optimum in [-80,80]^d.
    {target_optimum, _key} =
      Nx.Random.uniform(Nx.Random.key(12), -80, 80, shape: {d})

    # Compute shift so that the overall optimum is:
    # shift + unshifted_optimum = target_optimum  =>  shift = target_optimum - unshifted_optimum
    shift_tensor = Nx.subtract(target_optimum, unshifted_optimum)

    wrapped_fun =
      Nx.Defn.jit(fn x, shift ->
        # x: tensor with shape {n, d}
        # Broadcast shift to {1, d} before subtraction.
        bshift = Nx.new_axis(shift, 0)
        x_shifted = Nx.subtract(x, bshift)
        orig_fun.(x_shifted)
      end)

    merged = merged |> Map.put(:fun, wrapped_fun) |> Map.put(:shift, shift_tensor)
    struct(__MODULE__, merged)
  end

  @doc """
  Returns the overall optimal point of the wrapped function.

  By design, since:

      optimal_point = shift + optimum

  and shift is computed to force this sum to be equal to a random value drawn from [-80,80]^d,
  the returned optimal point always resides in [-80,80]^d.
  """
  @spec optimal_point(t()) :: Nx.Tensor.t()
  def optimal_point(%__MODULE__{shift: shift, optimum: optimum}) do
    Nx.add(shift, optimum)
  end
end
