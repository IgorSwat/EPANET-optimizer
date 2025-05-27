defmodule Hyperparameters do
  @type t :: %Hyperparameters{
          p_min: float(),
          p_max: float(),
          tau: float(),
          f_min: float(),
          f_max: float(),
          a0: float(),
          a1: float(),
          a2: float(),
          n: integer(),
          rand_fun: (() -> float()),
          mu: float() | nil,
          f: float() | nil
        }

  defstruct p_min: 0.5,
            p_max: 1.5,
            tau: 4.125,
            f_min: 0.07,
            f_max: 0.75,
            a0: 6.25,
            a1: 100,
            a2: 0.0005,
            n: 100,
            rand_fun: &:rand.uniform/0,
            mu: nil,
            f: nil
  
  defp compute_mu(tau) do
    2 / (abs(2 - tau - :math.sqrt(tau * tau - 4 * tau)))
  end

  defp compute_f(f_max, f_min) do
    f_min + (f_max - f_min) / (f_max + f_min)
  end
  
  @spec new(map()) :: t()
  def new(opts \\ %{}) do
    tau = Map.get(opts, :tau, %__MODULE__{}.tau)
    f_max = Map.get(opts, :f_max, %__MODULE__{}.f_max)
    f_min = Map.get(opts, :f_min, %__MODULE__{}.f_min)


    %__MODULE__{}
    |> struct(opts)
    |> Map.update!(:mu, fn _ -> compute_mu(tau) end)
    |> Map.update!(:f, fn _ -> compute_f(f_max, f_min) end)
  end


end
