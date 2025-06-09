defmodule SharedMacro do
  require Nx
  defmacro __using__(_) do
    quote do
      @d 10
      @n 100
      def inf, do: Nx.tensor(1.0e10)
      @pi :math.pi()
    end
  end
end
