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

	def runs(x, max_iterations \\ 100)

	def runs(0, _) do
	  {:ok}
	end

	def runs(x, max_iterations) when x > 0 do
	  OptimizerRunner.test(max_iterations)
	  OptimizerRunner.runs(x - 1, max_iterations)
	end

  def test(max_iterations \\ 1000) do
    hyperparams = Hyperparameters.new()
    problem = Problem.new(%{
      d: WhiteSharkOptimizer.d(),
      name: "Rosenbrock",
      fun: &Rosenbrock.evaluate_nx_matrix_defn/1,
      fun: &Rastrigin.evaluate_nx_matrix_defn/1,
      fun: &Levy.evaluate_nx_matrix_defn/1,
      fun: &Zakharov.evaluate_nx_matrix_defn/1,
      fun: &Schwefel.evaluate_nx_matrix_defn/1,
      minimize: true
    })

    wso = WhiteSharkOptimizer.new(problem, hyperparams, %{verbose: true, key: Nx.Random.key(12), max_iterations: max_iterations})
    start_time = System.monotonic_time()
    wso = WhiteSharkOptimizer.run(wso)
    end_time = System.monotonic_time()
    start_time_ms = System.convert_time_unit(start_time, :native, :millisecond)
    end_time_ms = System.convert_time_unit(end_time, :native, :millisecond)

	  IO.puts("#{(end_time_ms - start_time_ms)} #{format_solution(wso.best_g_fitness)}")
  end

  def run(max_iterations \\ 100) do
    hyperparams = Hyperparameters.new()
    problem = Problem.new(%{
      d: WhiteSharkOptimizer.d(),
      name: "Rosenbrock",
      fun: &Rosenbrock.evaluate_nx_matrix_defn/1,
      minimize: true
    })

	  true_solution = Nx.tensor(List.duplicate(3.0, WhiteSharkOptimizer.n()))

    IO.puts("TRUE SOLUTION: #{format_solution(problem.fun.(true_solution))} at #{format_solution(true_solution)}")

    start_time = System.monotonic_time()
    wso = WhiteSharkOptimizer.new(problem, hyperparams, %{verbose: true, key: Nx.Random.key(12), max_iterations: max_iterations})
    wso = WhiteSharkOptimizer.run(wso)
    end_time = System.monotonic_time()

	IO.puts("WSO Elapsed time: #{(end_time - start_time) / 10000} ms")
    IO.puts("WSO SOLUTION: #{wso.best_g_fitness} at #{format_solution(wso.wgbestk)}")
    IO.puts("MAE: #{calculate_mae(wso.wgbestk, true_solution)}")
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
