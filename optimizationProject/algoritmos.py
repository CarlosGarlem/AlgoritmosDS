#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re


# ### First functions

# In[56]:


def evaluate_Fx(str_equ, x):
    x = float(x)
    strOut = str_equ.replace('x', '*(x)')
    strOut = strOut.replace('^', '**')
    
    result = eval(strOut)
    return result


# In[3]:


def finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = '(' +  str_equ + ')'
    
    f1 = f1.replace('x', '*(x + 2*h)')
    f1 = f1.replace('^', '**')
    
    f2 = f2.replace('x', '*(x + h)')
    f2 = f2.replace('^', '**')
    
    f3 = f3.replace('x', '*(x)')
    f3 = f3.replace('^', '**')
      
    strOut = '(' + f1 + ' - 4 * ' + f2 + ' + 3 *' + f3 + ')' + ' / (2 * h)'
    result = eval(strOut)
    return result


# 
# ## Lab1 Functions

# #### R1 functions

# In[4]:


def center_finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = '(' +  str_equ + ')'
    
    f1 = f1.replace('x', '*(x + h)')
    f1 = f1.replace('^', '**')
    
    f2 = f2.replace('x', '*(x - h)')
    f2 = f2.replace('^', '**')
      
    strOut = '(' + f1 + ' - ' + f2 + ')' + ' / (2 * h)'
    #print(strOut)
    result = eval(strOut)
    return result


# In[5]:


def progressive_finite_derivative(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = '(' +  str_equ + ')'
   
    f1 = f1.replace('x', '*(x)')
    f1 = f1.replace('^', '**')
    
    f2 = f2.replace('x', '*(x + h)')
    f2 = f2.replace('^', '**')
    
    f3 = f3.replace('x', '*(x + 2 * h)')
    f3 = f3.replace('^', '**')
      
    strOut = '( -3 * ' + f1 + ' + 4 * ' + f2 + ' - ' + f3 + ')' + ' / (2 * h)'
    #print(strOut)
    result = eval(strOut)
    return result


# In[6]:


def center_finite_derivative_2(str_equ, x, h):
    x = float(x)
    h = float(h)
    f1 = f2 = f3 = f4 = '(' +  str_equ + ')'
    
    f1 = f1.replace('x', '*(x + h)')
    f1 = f1.replace('^', '**')
    
    f2 = f2.replace('x', '*(x - h)')
    f2 = f2.replace('^', '**')
    
    f3 = f3.replace('x', '*(x + 2 * h)')
    f3 = f3.replace('^', '**')
    
    f4 = f4.replace('x', '*(x - 2 * h)')
    f4 = f4.replace('^', '**')
      
    strOut = '(' + f4 + ' - 8 * ' + f2 + ' + 8 * ' + f1 + ' - ' + f3 + ')' + ' / (12 * h)'
    #print(strOut)
    result = eval(strOut)
    return result


# #### R2 Functions

# In[10]:


def center_finite_derivative_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
    
    str_parciales = []
    for var in ['x', 'y']:
        equ = str_equ.replace('y', '*(y)') if var == 'x' else str_equ.replace('x', '*(x)')
        f1 = f2 = '(' +  equ + ')'
 
        f1 = f1.replace(var, '*(' + var + ' + h)')
        f1 = f1.replace('^', '**')
    
        f2 = f2.replace(var, '*(' + var + ' - h)')
        f2 = f2.replace('^', '**')
      
        strOut = '(' + f1 + ' - ' + f2 + ')' + ' / (2 * h)'
        str_parciales.append(strOut)

        
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return result


# In[11]:


def progressive_finite_derivative_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
     
    str_parciales = []
    for var in ['x', 'y']:
        equ = str_equ.replace('y', '*(y)') if var == 'x' else str_equ.replace('x', '*(x)')
        f1 = f2 = f3 = '(' +  equ + ')'
   
        f1 = f1.replace(var, '*(' + var + ')')
        f1 = f1.replace('^', '**')

        f2 = f2.replace(var, '*(' + var + ' + h)')
        f2 = f2.replace('^', '**')

        f3 = f3.replace(var, '*(' + var + ' + 2 * h)')
        f3 = f3.replace('^', '**')
        
        strOut = '(-3 * ' + f1 + ' + 4 * ' + f2 + ' - ' + f3 + ')' + ' / (2 * h)'
        str_parciales.append(strOut)
        
        
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return result


# In[12]:


def center_finite_derivative_2_r2(str_equ, p, h):
    x = float(p[0])
    y = float(p[1])
    h = float(h)
     
    str_parciales = []
    for var in ['x', 'y']:
        equ = str_equ.replace('y', '*(y)') if var == 'x' else str_equ.replace('x', '*(x)')
        f1 = f2 = f3 = f4 = '(' +  equ + ')'
    
        f1 = f1.replace(var, '*(' + var +' + h)')
        f1 = f1.replace('^', '**')

        f2 = f2.replace(var, '*(' + var + ' - h)')
        f2 = f2.replace('^', '**')

        f3 = f3.replace(var, '*(' + var + ' + 2 * h)')
        f3 = f3.replace('^', '**')

        f4 = f4.replace(var, '*(' + var +' - 2 * h)')
        f4 = f4.replace('^', '**')
        strOut = '(' + f4 + ' - 8 * ' + f2 + ' + 8 * ' + f1 + ' - ' + f3 + ')' + ' / (12 * h)'
        str_parciales.append(strOut)
      
    
    #[print(parcial) for parcial in str_parciales]
    result = [eval(parcial, {}, {'x': x, 'y': y, 'h': h}) for parcial in str_parciales]
    return result



# ## Lab2 Functions

# In[57]:


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


# In[58]:


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


