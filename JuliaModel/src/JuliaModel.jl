module JuliaModel

import BeforeIT

function run_simulation(parameters, initial_conditions, T)
    model = BeforeIT.Model(parameters, initial_conditions)
    data = BeforeIT.run!(model, T)
    gdp = data.data.real_gdp
    return gdp
end

function get_parameters() 
    return BeforeIT.AUSTRIA2010Q1.parameters
end
function get_initial_conditions()
    return BeforeIT.AUSTRIA2010Q1.initial_conditions
end
end # module JuliaModel

