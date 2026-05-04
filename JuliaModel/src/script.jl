import BeforeIT as Bit

using Statistics

initial = Bit.AUSTRIA2010Q1.initial_conditions
parameters = Bit.AUSTRIA2010Q1.parameters



T = 40 
model = Bit.Model(parameters, initial)
data = Bit.run!(model, T)

firms_eq = [()]