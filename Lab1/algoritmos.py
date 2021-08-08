#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import re


# ### First functions

# In[2]:


#Evaluación REGEX
def evaluate_Fx(str_equ, valX):
    x = valX
    #strOut = str_equ
    strOut = str_equ.replace('x', '*(x)')
    strOut = strOut.replace('^', '**')
    out = eval(strOut)
    print(strOut)
    return out


# In[3]:


#Deferencias finitas para derivadas
def evaluate_derivate_fx(str_equ, x, h):
    x = float(x)
    h = float(h)
    strOut = str_equ.replace("x", '*(x + h)')
    strOut = strOut.replace("^", "**")
    strOut = "-4*(" + strOut + ")"
    out = eval(strOut)
  
    strOut = str_equ.replace("x", '*(x + 2*h)')
    strOut = strOut.replace("^", "**")
    out = out + eval(strOut)
  
    strOut = str_equ.replace("x", '*(x)')
    strOut = strOut.replace("^", "**")
    strOut = "3*(" + strOut + ")"
    out = out + eval(strOut)
  
    out = -out/(2*h)
    print(out)
    return out


# In[4]:


#Resolvedor de Newton
def newtonSolverX(x0, f_x, eps):
    x0 = float(x0)
    eps = float(eps)
    xn = x0
    error = 1
    arrayIters = []
    arrayF_x = []
    arrayf_x = []
    arrayXn = []
    arrayErr = []
  
    i = 0
    h = 0.000001
    while(error > eps):
        print("...")
        x_n1 = xn - (evaluate_Fx(f_x, xn)/evaluate_derivate_fx(f_x, xn, h))
        error = abs(x_n1 - xn)
        i += 1
        xn = x_n1
        arrayIters.append(i)
        arrayXn.append(xn)
        arrayErr.append(error)
        solution = [i, xn, error]

    print("Finalizo...")
    
    TableOut = pandas.DataFrame({'Iter':arrayIters, 'Xn':arrayXn, 'Error': arrayErr})
    return TableOut


# In[5]:


def add(a, b):
    a = int(a)
    b = int(b)
    resultado = a + b
    return "El resultado es: " + str(resultado)


# ### Lab1 Functions

# In[6]:


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


# In[7]:


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


# In[8]:


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



# ### Lab2 Functions

# In[12]:


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


# In[13]:


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


# In[14]:


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
