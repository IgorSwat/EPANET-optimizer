defmodule OptimizerRunner do
  require WhiteSharkOptimizer
  require Rosenbrock
  require Hyperparameters
  require Problem
  
	def runs(x, n \\ 100, max_iterations \\ 100, d \\ 10)

	def runs(0, _, _, _) do
	  {:ok}
	end

	def runs(x, n, max_iterations, d) when x > 0 do
	  OptimizerRunner.test(n, max_iterations, d)
	  OptimizerRunner.runs(x - 1, n, max_iterations, d)
	end

  def test(n \\ 100, max_iterations \\ 100, d \\ 10) do
    hyperparams = Hyperparameters.new(%{n: n})
    problem = Problem.new(%{
      d: d,
      name: "Rosenbrock",
      fun: &Rosenbrock.evaluate_nx/1,
      minimize: true
    })

	# true_solution = Nx.tensor(List.duplicate(3.0, d))

    # IO.puts("TRUE SOLUTION: #{format_solution(problem.fun.(true_solution))} at #{format_solution(true_solution)}")

    start_time = System.monotonic_time()
    wso = WhiteSharkOptimizer.new(problem, hyperparams, %{verbose: true, key: Nx.Random.key(12), max_iterations: max_iterations})
    wso = WhiteSharkOptimizer.run(wso)
    end_time = System.monotonic_time()

	IO.puts("#{(end_time - start_time) / 10000} #{format_solution(wso.best_g_fitness)}")
  end
	
  def run(n \\ 100, max_iterations \\ 100, d \\ 10) do
    hyperparams = Hyperparameters.new(%{n: n})
    problem = Problem.new(%{
      d: d,
      name: "Rosenbrock",
      fun: &Rosenbrock.evaluate_nx/1,
      minimize: true
    })

	true_solution = Nx.tensor(List.duplicate(3.0, d))

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
