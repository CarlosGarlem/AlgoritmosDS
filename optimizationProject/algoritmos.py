import pandas as pd
import numpy as np
import math
import pickle
import re


# ### Auxiliary functions

# In[2]:


def format_equation(str_equ, exp = 'x', var = 'x'):
    strOut = re.sub(r"{}".format(var), "({})".format(var), str_equ)
    strOut = re.sub(r"(?<=[A-Za-z0-9\)])(\({}\))".format(var), "*({})".format(var), strOut)
    strOut = re.sub(r"(?<=[0-9])\(", "*(", strOut)
    strOut = re.sub(r"(\^\({}\))".format(var), "**({})".format(var), strOut)
    strOut = re.sub(r"(\({}\))".format(var), "({})".format(exp), strOut)
    strOut = re.sub(r"\^", "**", strOut)
    
    strOut = re.sub(r"e", "math.e", strOut)
    strOut = re.sub(r"pi", "math.pi", strOut)
    strOut = re.sub(r"sin", "math.sin", strOut)
    strOut = re.sub(r"cos", "math.cos", strOut)
    strOut = re.sub(r"tan", "math.tan", strOut)
    strOut = re.sub(r"(?<=[A-Za-z0-9\)])(math.)", "*math.", strOut)
    return strOut


# In[3]:


def format_secondVar(str_equ, var):
    strOut = re.sub(r"(?<=[A-Za-z0-9\)])({})".format(var), "*({})".format(var), str_equ)
    return strOut



# In[7]:


def evaluate_Fx(str_equ, x):
    x = float(x)
    strOut = format_equation(str_equ, 'x')
    result = eval(strOut)
    return result


# In[8]:


def finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = '(' +  str_equ + ')'
    
    f1 = format_equation(f1, 'x + 2*h')
    f2 = format_equation(f2, 'x + h')
    f3 = format_equation(f3, 'x')
      
    strOut = '(' + f1 + ' - 4 * ' + f2 + ' + 3 *' + f3 + ')' + ' / (2 * h)'
    result = eval(strOut)
    return result


# ## Lab1 Functions (derivatives)

# #### R1 functions

# In[9]:


def center_finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = '(' +  str_equ + ')'
    
    f1 = format_equation(f1, 'x + h')
    f2 = format_equation(f2, 'x - h')
      
    strOut = '(' + f1 + ' - ' + f2 + ')' + ' / (2 * h)'
    result = eval(strOut)
    return np.array(result, dtype = np.float32)


# In[10]:


def progressive_finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = '(' +  str_equ + ')'
   
    f1 = format_equation(f1, 'x')
    f2 = format_equation(f2, 'x + h')
    f3 = format_equation(f3, 'x + 2*h')
      
    strOut = '( -3 * ' + f1 + ' + 4 * ' + f2 + ' - ' + f3 + ')' + ' / (2 * h)'
    result = eval(strOut)
    return np.array(result, dtype = np.float32)


# In[11]:


def center_finite_derivative_2(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = f4 = '(' +  str_equ + ')'
    
    f1 = format_equation(f1, 'x + h')
    f2 = format_equation(f2, 'x - h')   
    f3 = format_equation(f3, 'x + 2*h')   
    f4 = format_equation(f4, 'x - 2*h')   
      
    strOut = '(' + f4 + ' - 8 * ' + f2 + ' + 8 * ' + f1 + ' - ' + f3 + ')' + ' / (12 * h)'
    result = eval(strOut)
    return np.array(result, dtype = np.float32)



# #### R2 Functions

# In[12]:


def center_finite_derivative_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
    
    str_parciales = []
    for var in ['x', 'y']:
        equ = format_secondVar(str_equ, 'y') if var == 'x' else format_secondVar(str_equ, 'x')
        f1 = f2 = '(' +  equ + ')'
 
        f1 = format_equation(f1, var + ' + h', var)   
        f2 = format_equation(f2, var + ' - h', var)   
      
        strOut = '(' + f1 + ' - ' + f2 + ')' + ' / (2 * h)'
        str_parciales.append(strOut)
    
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return np.array(result, dtype = np.float32)


# In[13]:


def progressive_finite_derivative_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
     
    str_parciales = []
    for var in ['x', 'y']:
        equ = format_secondVar(str_equ, 'y') if var == 'x' else format_secondVar(str_equ, 'x')
        f1 = f2 = f3 = '(' +  equ + ')'
   
        f1 = format_equation(f1, var, var)   
        f2 = format_equation(f2, var + '+ h', var)   
        f3 = format_equation(f3, var + ' + 2*h', var)   
        
        strOut = '(-3 * ' + f1 + ' + 4 * ' + f2 + ' - ' + f3 + ')' + ' / (2 * h)'
        str_parciales.append(strOut)
        
        
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return np.array(result, dtype = np.float32)


# In[14]:


def center_finite_derivative_2_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
     
    str_parciales = []
    for var in ['x', 'y']:
        equ = format_secondVar(str_equ, 'y') if var == 'x' else format_secondVar(str_equ, 'x')
        f1 = f2 = f3 = f4 = '(' +  equ + ')'
    
        f1 = format_equation(f1, var + ' + h', var)   
        f2 = format_equation(f2, var + ' - h', var)   
        f3 = format_equation(f3, var + ' + 2*h', var)   
        f4 = format_equation(f4, var + ' - 2*h', var)   
        
        strOut = '(' + f4 + ' - 8 * ' + f2 + ' + 8 * ' + f1 + ' - ' + f3 + ')' + ' / (12 * h)'
        str_parciales.append(strOut)     
    
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return np.array(result, dtype = np.float32)


# ### Any R Space Derivative Function

# In[15]:


def center_finite_derivative_2_full(str_equ, p, h):
    x = np.array(p, dtype = np.float32)
    all_vars = ['x_{}'.format(i) for i in range(0, len(x))]
    vars_dict = {var: x_val for (var, x_val) in zip(all_vars, x)}
    vars_dict['h'] = h
    
    str_parciales = []
    equ = str_equ
    for i in range(0, len(x)):
        var = 'x_{}'.format(i)
        for incognita in all_vars:
            if incognita != var:
                equ = format_secondVar(equ, incognita)
        f1 = f2 = f3 = f4 = '(' +  equ + ')'
    
        f1 = format_equation(f1, var + ' + h', var)   
        f2 = format_equation(f2, var + ' - h', var)   
        f3 = format_equation(f3, var + ' + 2*h', var)   
        f4 = format_equation(f4, var + ' - 2*h', var)   
        
        strOut = '(' + f4 + ' - 8 * ' + f2 + ' + 8 * ' + f1 + ' - ' + f3 + ')' + ' / (12 * h)'
        str_parciales.append(strOut)     
    
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, vars_dict) for parcial in str_parciales]
    return np.array(result, dtype = np.float32)


# ## Lab2 Functions (ceros)

# In[16]:


def metodo_biseccion(str_equ, interval, k_max, epsilon):
    k = 0
    a = float(interval[0])
    b = float(interval[1])
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Error': []}
    
    x_k = (a + b) / 2    
    while k < k_max and error > epsilon:
        Fa = evaluate_Fx(str_equ, a)
        Fx_k = evaluate_Fx(str_equ, x_k)
        if (Fa * Fx_k) < 0:
            b = x_k
        else:
            a = x_k
        
        k += 1
        x_k = (a + b) / 2
        error = abs(Fx_k)
        
        data['Iter'].append(k)
        data['Xn'].append(x_k)
        data['Error'].append(error)
        
    results = pd.DataFrame(data)
    return results


# In[17]:


def metodo_newton(str_equ, x_0, k_max, epsilon):
    k = 0
    x_k = x_0
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Error': []}
    
    while k < k_max and error > epsilon:
        Fx_k = evaluate_Fx(str_equ, x_k)
        dev1_Fx_k = center_finite_derivative_2(str_equ, x_k, 0.00001)
        
        x_k1 = x_k - (Fx_k / dev1_Fx_k)
        x_k = x_k1
        k += 1
        error = abs(Fx_k)
        
        data['Iter'].append(k)
        data['Xn'].append(x_k)
        data['Error'].append(error)
        
    results = pd.DataFrame(data)
    return results



# ## Lab3 Functions (GD)

# In[113]:


def parseInput(x, reshape = False):
    if reshape:
        rows = len(x)
        x = np.array(x, dtype = 'float').reshape(rows, 1)
    else:
        x = np.array(x, dtype = 'float')

    #return x.shape
    return x


# In[114]:


def getLearningRate(opt, k, gradient = None, Q = None, alpha = 0.001):
    if opt == 'Exacto':
        lr = (np.linalg.norm(gradient, 2)**2) / np.matmul(np.matmul(gradient.T, Q), gradient)
        lr = lr[0, 0]
    elif opt == 'Constante':
        lr = alpha
    elif opt == 'Variable':
        lr = 1/k
    else:
        lr = 1
        
    return lr


# In[115]:


def gradient_descent_QP(x_0, Q, c, kmax, epsilon, lr_type, alpha = 0.001):
    k = 0
    x_k = parseInput(x_0, reshape = True)
    c = parseInput(c, reshape = True)
    Q = parseInput(Q)
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Pk': [], 'Error': []}
      
    while k < kmax and error > epsilon:
        gradient = np.matmul(Q, x_k) + c
        lr = getLearningRate(lr_type, k + 1, gradient, Q, alpha)
        x_k1 = x_k - (lr * gradient)
        
        x_k = x_k1
        k += 1
        error = np.linalg.norm(gradient, 2)
        
        data['Iter'].append(k)
        data['Xn'].append(x_k.round(7))
        data['Pk'].append(-1 * gradient.round(7))
        data['Error'].append(error.round(7))     
  
    results = pd.DataFrame(data)
    return results


# In[100]:


def getRosenbrockGradient(x0):
    x = x0[0, 0]
    y = x0[1, 0]
    g1 = eval('400*(x**3) + 2*(x) - 400*(x)*(y) - 2', {}, {'x': x, 'y': y})
    g2 = eval('-200*(x**2) + 200*(y)', {}, {'x': x, 'y': y})
    gradient = np.array([g1, g2], dtype = 'float').reshape(2,1)
    
    return gradient


# In[101]:


def rosenbrock_gd(x_0, kmax, epsilon, lr):
    k = 0
    x_k = parseInput(x_0, reshape = True)
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Pk': [], 'Error': []}
      
    while k < kmax and error > epsilon:
        gradient = getRosenbrockGradient(x_k)
        x_k1 = x_k - (lr * gradient)

        x_k = x_k1
        k += 1
        error = np.linalg.norm(gradient, 2)
        
        data['Iter'].append(k)
        data['Xn'].append(x_k.round(7))
        data['Pk'].append(-1 * gradient.round(7))
        data['Error'].append(error.round(7))
    
    results = pd.DataFrame(data)
    return results


   
# ### Lab4 (GD Variants and Newton)

# ### GD Variants

# In[165]:
def generateData(d, n, path):
    A = np.random.normal(0, 1, size = (n,d))
    x_true = np.random.normal(0, 1, size = (d,1))
    b = A.dot(x_true) + np.random.normal(0, 0.5, size = (n,1))
    data = {'x_true': x_true, 'b': b, 'A': A}

    with open(path, 'wb') as pickle_out:
        pickle.dump(data, pickle_out)

    return 'Data store successfully on path ' + path


# In[38]:
def getData(filepath):
    with open(filepath,"rb") as pickle_in:
        data = pickle.load(pickle_in)

    A = data['A']
    b = data['b']
    x_true = data['x_true']

    return A, b, x_true


# ##### Parte 1 - Close solution
# In[143]:
def getCloseSolution(path):
    A, b, x_true = getData(path)
    inverse = np.linalg.inv(np.matmul(A.T, A))
    x = np.matmul(np.matmul(inverse, A.T), b)
    f_x = np.sum((np.matmul(A, x) - b)**2)
    e_x = np.linalg.norm(x - x_true)

    gradient = np.matmul(np.matmul(A.T, A), x) - np.matmul(A.T, b)
    error = np.linalg.norm(gradient)
    results = pd.DataFrame({'Iter': 1, 'Xn': [x], 'Error': error, 'F*': f_x, 'E_n': e_x})

    return results


# ##### Parte 2 - GD
# In[159]:
def computeGD(x_0, A, b, x_true, kmax, lr, mb_size, epsilon):
    k = 0
    x_k = x_0
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Pk': [], 'Error': [], 'F*': [], 'E_n': []}

    mat = np.hstack((A,b))
    while k < kmax and error > epsilon:
        np.random.shuffle(mat) #in-place shuffle
        iters = A.shape[0] // mb_size
        for i in range(0, iters):
            start = i * mb_size
            end = (1+i) * mb_size
            A_mb = mat[start:end, :-1]
            b_mb = mat[start:end, -1]
            b_mb = np.expand_dims(b_mb, axis = 1)

            gradient = np.matmul(np.matmul(A_mb.T, A_mb), x_k) - np.matmul(A_mb.T, b_mb)
            x_k1 = x_k - (lr * gradient)
            x_k = x_k1

        k += 1
        f_k = np.sum((np.matmul(A, x_k) - b)**2)
        e_k = np.linalg.norm(x_k - x_true)
        error = np.linalg.norm(gradient)
        data['Iter'].append(k)
        data['Xn'].append(x_k.round(7))
        data['Pk'].append(-1 * gradient.round(7))
        data['Error'].append(error.round(7))
        data['F*'].append(f_k.round(7))
        data['E_n'].append(e_k.round(7))

    results = pd.DataFrame(data)
    return results


# In[142]:
def gdSolver(path, kmax, lr, mb_size, variant = 'GD', epsilon = 0.00000001):
    A, b, x_true = getData(path)
    x_0 = np.zeros_like(x_true)
    if variant == 'GD':
        df = computeGD(x_0, A, b, x_true, kmax, lr, A.shape[0], epsilon)
    elif variant == 'SGD':
        df = computeGD(x_0, A, b, x_true, kmax, lr, 1, epsilon)
    elif variant == 'MBGD':
        df = computeGD(x_0, A, b, x_true, kmax, lr, mb_size, epsilon)

    return df


# #### Metodo de Newton

# ##### Parte 1 - GD con Backtracking line search
# In[58]:
def evalRosenbrockFunction(x0):
    x = x0[0, 0]
    y = x0[1, 0]
    rsb_function = '100*((y-x**2)**2) + (1 - x)**2'
    result = eval(rsb_function, {}, {'x': x, 'y': y})
    return result


# In[59]:
def backTrackingLineSearch(x_0, lr, ro, c):
    x_k = x_0
    condition = True

    while condition:
        gradient = getRosenbrockGradient(x_k)
        x_k1 = x_k - (lr * gradient)

        fk_1 = evalRosenbrockFunction(x_k1)
        f_k = evalRosenbrockFunction(x_k)
        rhs = c * lr * np.matmul(gradient.T, -gradient)

        condition = (fk_1 > (f_k + rhs)) #the loop is the negated condition of the backtracking algorithm
        lr *= ro

    return lr


# In[60]:
def rosenbrock_backtracking(x_0, kmax, epsilon, alpha, lr_type = 'backtracking', ro = 0.5, c = 0.0001):
    k = 0
    x_k = parseInput(x_0, reshape = True)
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Pk': [], 'Error': []}

    if lr_type == 'backtracking':
        lr = backTrackingLineSearch(x_k, alpha, ro, c)
    else: #else it would be constant
        lr = alpha

    while k < kmax and error > epsilon:
        gradient = getRosenbrockGradient(x_k)
        x_k1 = x_k - (lr * gradient)

        x_k = x_k1
        k += 1
        error = np.linalg.norm(gradient)

        data['Iter'].append(k)
        data['Xn'].append(x_k.round(7))
        data['Pk'].append(-1 * gradient.round(7))
        data['Error'].append(error.round(7))

    results = pd.DataFrame(data)
    return results


# ##### Parte 2 - Metodo de newton con Backtracking line search
# In[63]:
def getRosenbrockHessian(x0):
    x = x0[0, 0]
    y = x0[1, 0]
    g1 = eval('1200*(x**2) - 400*(y) + 2', {}, {'x': x, 'y': y})
    g2 = eval('-400*(x)', {}, {'x': x})
    hessian = np.array([g1, g2, g2, 200], dtype = 'float').reshape(2,2)

    return hessian


# In[64]:
def newton_optimization(x_0, kmax, epsilon, alpha, lr_type = 'backtracking', ro = 0.5, c = 0.0001):
    k = 0
    x_k = parseInput(x_0, reshape = True)
    error = float('inf')
    data = {'Iter': [], 'Xn': [], 'Pk': [], 'Error': []}

    if lr_type == 'backtracking':
        lr = backTrackingLineSearch(x_k, alpha, ro, c)
    else: #else it would be constant
        lr = alpha

    while k < kmax and error > epsilon:
        gradient = getRosenbrockGradient(x_k)
        hessian = getRosenbrockHessian(x_k)
        p_k = -1 * np.matmul(np.linalg.inv(hessian), gradient)
        x_k1 = x_k + (lr * p_k)

        x_k = x_k1
        k += 1
        error = np.linalg.norm(gradient)

        data['Iter'].append(k)
        data['Xn'].append(x_k.round(7))
        data['Pk'].append(p_k.round(7))
        data['Error'].append(error.round(7))

    results = pd.DataFrame(data)
    return results