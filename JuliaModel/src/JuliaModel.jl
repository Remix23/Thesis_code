module JuliaModel

import BeforeIT

function run_simulation(parameters, initial_conditions, T)
    if Threads.nthreads() > 1
        parallel = true
    else
        parallel = false
    end
    model = BeforeIT.Model(parameters, initial_conditions)
    data = BeforeIT.run!(model, T, parallel=parallel)
    gdp = data.data.real_gdp
    return gdp
end

function run_monte_carlo(parameters, initial_conditions, T, num_simulations)
    models = [BeforeIT.Model(parameters, initial_conditions) for _ in 1:num_simulations]
    data = BeforeIT.ensemblerun!(models, T, parallel=true)
    gdps = [d.data.real_gdp for d in data]
    gdp = mean(gdps, dims=1)
    return gdp
end

function run_for_different_parameters(parameters_list, initial_conditions, T)
    models = [BeforeIT.Model(parameters, initial_conditions) for parameters in parameters_list]
    data = BeforeIT.ensemblerun!(models, T, parallel=true)
    gdps = [d.data.real_gdp for d in data]
    return gdps
end

function get_parameters() 
    println(Threads.nthreads())
    return BeforeIT.AUSTRIA2010Q1.parameters
end
function get_initial_conditions()
    return BeforeIT.AUSTRIA2010Q1.initial_conditions
end
end # module JuliaModel

