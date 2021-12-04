from sympy.abc import *
from sympy import ordered, hessian, Matrix
import numpy as np

def Newton_Optimizer_UN(f, initial_val=[0,0]):
    #Fetch Variables and declare gradient()
    variables= list(ordered(f.free_symbols))
    gradient = lambda f, variables: Matrix([f]).jacobian(variables)
    #Make sure inital values are sufficient
    if len(initial_val) != len(variables):
        initial_val = [0]*len(initial_val)
    
    #Compute Gradient and Hessian
    gradMat = (gradient(f, variables)).T   #Transposes the Matrix computed by a 1x2 Jacobian
    hessMat = hessian(f, variables)

    #Compute Gradient at given points
    gradMat_at_point = gradMat.subs([(variables[values],initial_val[values]) for values in range(len(initial_val))])
    
    optimized_parameters = Matrix(len(variables),1, [values for values in initial_val])
    
    while(np.prod(gradMat_at_point) != 0):  #Continue till gradient becomes 0
        #Compute Hessian at initial points:
        hessMat_at_point = hessMat.subs([(variables[values],optimized_parameters[values]) for values in range(len(optimized_parameters))])
        
        #Xop = Xi - inverse(Hessian_of(f))*Nabla_of(f)
        optimized_parameters = optimized_parameters -((hessMat_at_point.inv())*(gradMat_at_point))

        #Recomputing using new guesses
        gradMat_at_point = gradMat.subs([(variables[values],optimized_parameters[values]) for values in range(len(optimized_parameters))])
        
    #Positive Definite Test:
    op_T = optimized_parameters.T
    posDefTest = op_T*hessMat_at_point*optimized_parameters #[Xt]*[H]*[X] > 0
    if(posDefTest[0] > 0):
        print("Hessian is Positive Definite. ")
        print("Minima at: ")
    else:
        print("Hessian is Not-Positive Definite. ")
        print("Maxima at: ")

    for parameters in range(len(variables)):
        print("%s = %s" % (variables[parameters],optimized_parameters[parameters]))
    return optimized_parameters


#EXAMPLE:

f = 719*m + 1469*n - 0.01*n**2 - 0.01*m**2 - 0.007*m*n - 1000000000
optimized_parameters = Newton_Optimizer_UN(f)