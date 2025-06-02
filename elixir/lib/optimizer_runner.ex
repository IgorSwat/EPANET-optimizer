defmodule OptimizerRunner do
  require WhiteSharkOptimizer
  require Rosenbrock
  require Rastrigin
  require Levy
  require Zakharov
  require Schwefel
  require Hyperparameters
  require Problem
  require SharedMacro
  require BentCigar

  @doc """
  Runs each optimization function in a series of tests using custom parameters for
  dimension (`d`) and bounds (`l` and `u`).
  """
  def run_all(iterations, max_iterations \\ 1000) when iterations > 0 do
    # A list of function configurations. You can customize d, l, and u for each function.
    d = WhiteSharkOptimizer.d()
    functions = [
      %{
        name: "Rosenbrock",
        fun: &Rosenbrock.evaluate_nx_matrix_defn/1,
        d: d,
        l: -100,
        u: 100
      },
      %{
        name: "Rastrigin",
        fun: &Rastrigin.evaluate_nx_matrix_defn/1,
        d: d,
        l: -100,
        u: 100
      },
      %{
        name: "Levy",
        fun: &Levy.evaluate_nx_matrix_defn/1,
        d: d,
        l: -100,
        u: 100
      },
      %{
        name: "Zakharov",
        fun: &Zakharov.evaluate_nx_matrix_defn/1,
        d: d,
        l: -100,
        u: 100
      },
      %{
        name: "Schwefel",
        fun: &Schwefel.evaluate_nx_matrix_defn/1,
        d: d,
        l: -500,
        u: 500
      },
      %{name: "BentCigar",
        fun: &BentCigar.evaluate_nx_matrix_defn/1,
        d: d,
        l: -100,
        u: 100
    }
    ]

    Enum.each(functions, fn func_config ->
      run_function(iterations, max_iterations, func_config)
    end)
  end

  # Recursively runs the test for a given function configuration.
  defp run_function(0, _max_iterations, _func_config), do: :ok

  defp run_function(n, max_iterations, %{name: name, fun: fun, d: d, l: l, u: u} = config) when n > 0 do
    test(max_iterations, name, fun, d, l, u)
    run_function(n - 1, max_iterations, config)
  end

  @doc """
  Sets up the optimization problem with the provided dimension and bounds.
  Then it runs the optimizer and prints the result.
  """
  def test(max_iterations \\ 1000, name, fun, d, l, u) do
    hyperparams = Hyperparameters.new()

    # Build the problem using the given dimension d.
    problem = Problem.new(%{
      d: d,
      name: name,
      fun: fun
    })

    # Create the optimizer using the custom lower (`l`) and upper (`u`) bounds.
    wso =
      WhiteSharkOptimizer.new(problem, hyperparams, %{
        verbose: true,
        key: Nx.Random.key(12),
        max_iterations: max_iterations,
        l: Nx.broadcast(l, {d}),
        u: Nx.broadcast(u, {d})
      })

    start_time = System.monotonic_time()
    wso = WhiteSharkOptimizer.run(wso)
    end_time = System.monotonic_time()

    start_time_ms = System.convert_time_unit(start_time, :native, :millisecond)
    end_time_ms = System.convert_time_unit(end_time, :native, :millisecond)

    IO.puts("NX #{max_iterations} #{WhiteSharkOptimizer.n()} #{problem.d} #{problem.name} #{(end_time_ms - start_time_ms)} #{format_solution(wso.best_g_fitness)}")
  end

  def run(max_iterations \\ 100) do
    hyperparams = Hyperparameters.new()
    problem = Problem.new(%{
      d: WhiteSharkOptimizer.d(),
      name: "Rosenbrock",
      fun: &Rosenbrock.evaluate_nx_matrix_defn/1,
      l: Nx.broadcast(-1, {WhiteSharkOptimizer.d()}),
      u: Nx.broadcast(1, {WhiteSharkOptimizer.d()})
    })

    start_time = System.monotonic_time()
    wso = WhiteSharkOptimizer.new(problem, hyperparams, %{verbose: false, key: Nx.Random.key(12), max_iterations: max_iterations})
    wso = WhiteSharkOptimizer.run(wso)
    end_time = System.monotonic_time()

	  IO.inspect("WSO Elapsed time: #{(end_time - start_time) / 10000} ms")
    IO.inspect("WSO SOLUTION: #{format_solution(wso.best_g_fitness)} at #{format_solution(wso.wgbestk)}")
    #IO.puts("MAE: #{calculate_mae(wso.wgbestk, true_solution)}")
  end

  defp format_solution(solution) do
    if Nx.rank(solution) == 0 do
      solution
      |> Nx.to_number()
      |> to_string()
    else
      solution
      |> Nx.to_list()
      |> List.flatten()
      |> Enum.map(&Float.to_string/1)
      |> Enum.join(" ")
    end
  end

  defp calculate_mae(predicted, true_solution) do
    predicted
    |> Nx.subtract(true_solution)
    |> Nx.abs()
    |> Nx.mean()
    |> Nx.to_number()
  end
end
