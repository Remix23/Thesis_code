module JuliaModel

import BeforeIT
using Dates

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
    return gdps
end

function run_for_different_parameters(parameters_list, initial_conditions, T)
    models = [BeforeIT.Model(parameters, initial_conditions) for parameters in parameters_list]
    data = BeforeIT.ensemblerun!(models, T, parallel=true)
    gdps = [d.data.real_gdp for d in data]
    return gdps
end

function get_real()
    cal = BeforeIT.ITALY_CALIBRATION
    d = cal.data["real_gdp_quarterly"]
    first = DateTime(1996, 3, 31)
    quarters = length(d)
    quarterly_dates = [first + Month(3 * i) for i in 0:(quarters - 1)]
    return [quarterly_dates, d]
end

function calibrate(year, month, day) 
    cal = BeforeIT.ITALY_CALIBRATION
    calibration_date = DateTime(year, month, day)
    parameters, initial_conditions = BeforeIT.get_params_and_initial_conditions(cal, calibration_date; scale = 0.0001)
    println("Calibrating model with date: ", calibration_date)
    return parameters, initial_conditions
end
end # module JuliaModel

