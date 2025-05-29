defmodule Problem do
  @derive {Nx.Container, containers: [:l, :u]}
  @enforce_keys [:d]
  defstruct name: nil,
            d: 10,
            fun: nil,
            l: nil,
            u: nil,
            minimize: true

  @type t :: %__MODULE__{
          name: String.t() | nil,
          d: integer(),
          fun: (Nx.Tensor.t() -> float()) | nil,
          l: Nx.Tensor.t() | nil,
          u: Nx.Tensor.t() | nil,
          minimize: boolean()
        }

  @spec new(map()) :: t()
  def new(%{d: d} = opts) do
    fun = case Map.get(opts, :minimize, true) do
      true -> Map.get(opts, :fun, nil)
      false -> fn tensor -> -Map.get(opts, :fun, nil).(tensor) end
    end

    computed_fields = %{
      l: Nx.broadcast(-10, {d}),
      u: Nx.broadcast(10, {d}),
      fun: fun
    }

    struct(__MODULE__, Map.merge(computed_fields, opts))
  end
end
