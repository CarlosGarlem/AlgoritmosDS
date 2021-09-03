library(shiny)
library(shinydashboard)
#library(DT)

dashboardPage(
    
    dashboardHeader(title = 'Algoritmos DS'),
    
    dashboardSidebar(
        sidebarMenu(
            menuItem('Diferenciacion Numerica en R', tabName = 'diffNum'),
            menuItem('Diferenciacion Numerica en R2', tabName = 'diffNumR2'),
            menuItem('Metodo Newton', tabName = 'ceros_newton'),
            menuItem('Metodo Biseccion', tabName = 'ceros_biseccion'),
            menuItem('GD problemas QP', tabName = 'gd_qp'),
            menuItem('GD Funcion Rosenbrock', tabName = 'gd_rosenbrock')
        )
    ),
    
    dashboardBody(
        tabItems(
            
            tabItem('diffNum',
                    h1('Diferenciacicion Numerica'),
                        
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            radioButtons('algorithms_diffnum',
                                        'Seleccione un algoritmo',
                                        choices = c('Diferencia Finita Centrada' = 'diffCenter',
                                                    'Diferencia Finita Progresiva' = 'diffProgressive',
                                                    'Diferencia Finita Centrada2' = 'diffCenter2'),
                                        selected = 'diffCenter'),
                            textInput('function_diffnum', 'Ingrese la Ecuacion'),
                            #strong('Ecuacion'),
                            #p('3x^4 - 2x^3 -4x^2 + 5x + 2'),
                            numericInput('x_diffnum', 'X0', value = 1),
                            numericInput('h_diffnum', 'h', value = 1),
                            actionButton('eval_diffnum', 'Evaluar Derivada')
                        ),
                        box(width = 4, solidHeader = TRUE,
                            tableOutput('diffnum_table')
                        )
                    )
            ),
            
            tabItem('diffNumR2',
                    h1('Diferenciacicion Numerica en R2'),
                    
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            radioButtons('algorithms_diffnum_r2',
                                         'Seleccione un algoritmo',
                                         choices = c('Diferencia Finita Centrada' = 'diffCenter',
                                                     'Diferencia Finita Progresiva' = 'diffProgressive',
                                                     'Diferencia Finita Centrada2' = 'diffCenter2'),
                                         selected = 'diffCenter'),
                            textInput('function_diffnum_r2', 'Ingrese la Ecuacion'),
                            textInput('x_diffnum_r2', 'P(X0,Y0)'),
                            numericInput('h_diffnum_r2', 'h', value = 1),
                            actionButton('eval_diffnum_r2', 'Evaluar Derivada')
                        ),
                        box(width = 4, solidHeader = TRUE,
                            tableOutput('diffnum_table_r2')
                        )
                    )
            ),
            
            tabItem('ceros_newton',
                    h1('Metodo de Newton'),
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            textInput('function_newton', 'Ingrese la Ecuacion'),
                            numericInput('newton_x0', 'X0', value = 1),
                            numericInput('newton_e', 'Tolerancia (e)', value = 0.001),
                            numericInput('newton_iters', 'Kmax', value = 10),
                            actionButton('eval_newton', 'Encontrar Ceros')
                        ),
                        box(width = 4, solidHeader = TRUE,
                            DT::dataTableOutput('newton_table')
                        )
                    )
            ),
            
            tabItem('ceros_biseccion',
                    h1('Metodo de biseccion'),
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            textInput('function_biseccion', 'Ingrese la Ecuacion'),
                            textInput('biseccion_interval', '[a,b]'),
                            numericInput('biseccion_e', 'Tolerancia (e)', value = 0.001),
                            numericInput('biseccion_iters', 'Kmax', value = 10),
                            actionButton('eval_biseccion', 'Encontrar Ceros')
                        ),
                        box(width = 4, solidHeader = TRUE,
                            DT::dataTableOutput('biseccion_table')
                        )
                    )
            ),
            
            
            tabItem('gd_qp',
                    h1('GD Problemas QP'),
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            textInput('qp_Qmatrix', 'Q', placeholder = '[[1,2],[3,4]]', value = '[[2,-1,0],[-1,2,-1],[0,-1,2]]'),
                            textInput('qp_Cvector', 'C', placeholder = '[1,2]', value = '[1,0,1]'),
                            textInput('qp_X0', 'X0', placeholder = '[1,2]', value = '[3,5,7]'),
                            numericInput('qp_epsilon', 'Tolerancia (e)', value = 0.000001),
                            numericInput('qp_iters', 'N', value = 30),
                            radioButtons('qp_rbuttons',
                                         'Tipo de Step Size',
                                         choices = c('Exacto', 'Constante', 'Variable'),
                                         selected = 'Exacto'),
                            numericInput('qp_lr', 'Learning Rate', value = 0.001),
                            actionButton('eval_qp', 'Evaluar GD')
                        ),
                        box(width = 8, solidHeader = TRUE,
                            DT::dataTableOutput('qp_table')
                        )
                    )
            ),
            
            tabItem('gd_rosenbrock',
                    h1('GD Funcion de Rosenbrock'),
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            textInput('rf_X0', 'X0', placeholder = '[1,2]', value = '[0, 0]'),
                            numericInput('rf_epsilon', 'Tolerancia (e)', value = 0.00000001),
                            numericInput('rf_iters', 'N', value = 1000),
                            numericInput('rf_lr', 'Learning Rate', value = 0.05),
                            actionButton('eval_rf', 'Evaluar GD')
                        ),
                        box(width = 8, solidHeader = TRUE,
                            DT::dataTableOutput('rf_table')
                        )
                    )
            )
            
            
        )
    )
)
