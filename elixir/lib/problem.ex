defmodule Problem do
  @type t :: %Problem{
          name: String.t() | nil,
          d: integer(),
          fun: (Nx.Tensor.t() -> float()) | nil,
          l: Nx.Tensor.t() | nil,
          u: Nx.Tensor.t() | nil,
          minimize: boolean(),
        }

  defstruct name: nil,
            d: nil,
            fun: nil,
            l: nil,
            u: nil,
            minimize: true

  @spec new(integer()) :: t()
  def new(opts \\ %{}) do
    d = Map.get(opts, :d, 3)
    fun = case Map.get(opts, :minimize, true) do
      
      true -> Map.get(opts, :fun, nil)
      false -> fn tensor -> -Map.get(opts, :fun, nil).(tensor) end
    end
    
    computed_fields = %{
      l: Nx.broadcast(-10, {d}),
      u: Nx.broadcast(10, {d}),
      fun: fun

    }
    %__MODULE__{}
    |> struct(Map.merge(computed_fields, opts))
    |> struct(opts)
      
  end
end

