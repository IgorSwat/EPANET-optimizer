defmodule SharedMacro do
  require Nx
  defmacro __using__(_) do
    quote do
      @d 1000
      @n 100
      @inf Nx.tensor(1.0e10)
      @pi :math.pi()
    end
  end
end
