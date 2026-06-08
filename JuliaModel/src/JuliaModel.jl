module JuliaModel


import BeforeIT_Modded
using Dates

function processSimulationOutput(data, keys)
    out = zeros(length(keys), size(data.real_gdp, 1))
    for (i, key) in enumerate(keys)
        s = Symbol(key)
        if key == "gdp_deflator"
            out[i, :] = data.nominal_gdp ./ data.real_gdp 
        else
            if !hasproperty(data, s)
                println("Key $(key) not found in model output.")
                continue
            end
            x = getfield(data, s)
            if ndims(x) != 1
                println("Warning: $(key) is not 1-dimensional, skipping.")
                continue
            end
            out[i, :] = x
        end
        
    end
    return out
end 

function run_simulation(parameters, initial_conditions, T, keys)
    if Threads.nthreads() > 1
        parallel = true
    else
        parallel = false
    end
    model = BeforeIT_Modded.Model(parameters, initial_conditions)
    data = BeforeIT_Modded.run!(model, T, parallel=parallel)
    return processSimulationOutput(data.data, keys)
end

function run_monte_carlo(parameters, initial_conditions, T, num_simulations, keys)
    models = [BeforeIT_Modded.Model(parameters, initial_conditions) for _ in 1:num_simulations]
    data = BeforeIT_Modded.ensemblerun!(models, T, parallel=true)
    out = zeros(num_simulations, length(keys), T + 1)
    for (i, d) in enumerate(data)
        out[i, :, :] = processSimulationOutput(d.data, keys)
    end
    return out
end

function run_for_different_parameters(parameters_list, initial_conditions, T, keys)
    models = [BeforeIT_Modded.Model(parameters, initial_conditions) for parameters in parameters_list]
    data = BeforeIT_Modded.ensemblerun!(models, T, parallel=true)
    out = zeros(length(parameters_list), length(keys), T + 1)
    for (i, d) in enumerate(data)
        out[i, :, :] = processSimulationOutput(d.data, keys)
    end
    return out
end

function get_real(keys)
    
    cal = BeforeIT_Modded.ITALY_CALIBRATION
    out = Dict()
    for key in keys
        name = key * "_quarterly"
        out[key] = cal.data[name]
    end
    first = DateTime(1996, 3, 31)
    quarters = length(out[keys[1]])
    quarterly_dates = [first + Month(3 * i) for i in 0:(quarters - 1)]
    
    return [out, quarterly_dates]
end

function calibrate(year, month, day) 
    cal = BeforeIT_Modded.ITALY_CALIBRATION
    calibration_date = DateTime(year, month, day)
    parameters, initial_conditions = BeforeIT_Modded.get_params_and_initial_conditions(cal, calibration_date; scale = 0.0001)
    println("Calibrating model with date: ", calibration_date)
    return parameters, initial_conditions
end
end # module JuliaModel

