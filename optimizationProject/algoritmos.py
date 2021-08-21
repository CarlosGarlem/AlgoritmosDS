# In[1]:


import pandas as pd
import math 
import re


# ### Auxiliary functions

# In[2]:


def format_equation(str_equ, exp = 'x', var = 'x'):
    strOut = re.sub(r"{}".format(var), "({})".format(var), str_equ)
    strOut = re.sub(r"(?<=[A-Za-z0-9\)])(\({}\))".format(var), "*({})".format(var), strOut)
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


# In[29]:


def format_secondVar(str_equ, var):
    strOut = re.sub(r"(?<=[A-Za-z0-9\)])({})".format(var), "*({})".format(var), str_equ)
    return strOut




# In[3]:


def evaluate_Fx(str_equ, x):
    x = float(x)
    strOut = format_equation(str_equ, 'x')
    result = eval(strOut)
    return result


# In[4]:


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


# 
# ## Lab1 Functions

# #### R1 functions

# In[5]:


def center_finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = '(' +  str_equ + ')'
    
    f1 = format_equation(f1, 'x + h')
    f2 = format_equation(f2, 'x - h')
      
    strOut = '(' + f1 + ' - ' + f2 + ')' + ' / (2 * h)'
    result = eval(strOut)
    return result


# In[6]:


def progressive_finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = '(' +  str_equ + ')'
   
    f1 = format_equation(f1, 'x')
    f2 = format_equation(f2, 'x + h')
    f3 = format_equation(f3, 'x + 2*h')
      
    strOut = '( -3 * ' + f1 + ' + 4 * ' + f2 + ' - ' + f3 + ')' + ' / (2 * h)'
    result = eval(strOut)
    return result


# In[7]:


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
    return result




# #### R2 Functions

# In[37]:


def center_finite_derivative_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
    
    str_parciales = []
    for var in ['x', 'y']:
        #equ = str_equ.replace('y', '*(y)') if var == 'x' else str_equ.replace('x', '*(x)')
        equ = format_secondVar(str_equ, 'y') if var == 'x' else format_secondVar(str_equ, 'x')
        f1 = f2 = '(' +  equ + ')'
 
        f1 = format_equation(f1, var + ' + h', var)   
        f2 = format_equation(f2, var + ' - h', var)   
      
        strOut = '(' + f1 + ' - ' + f2 + ')' + ' / (2 * h)'
        str_parciales.append(strOut)
    
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return result


# In[38]:


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
    return result


# In[39]:


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
    return result



# ## Lab2 Functions

# In[13]:


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


# In[14]:


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

