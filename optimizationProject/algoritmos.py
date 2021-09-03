import pandas as pd
import numpy as np
import math 
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