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
            menuItem('GD Funcion Rosenbrock', tabName = 'gd_rosenbrock'),
            menuItem('Generar Datos', tabName = 'generar_datos'),
            menuItem('GD Solver', tabName = 'gd_solver'),
            menuItem('Backtracking Rosenbrock', tabName = 'bt_solver')
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
                            textInput('qp_Qmatrix', 'Q', placeholder = '[[2,-1,0],[-1,2,-1],[0,-1,2]]'),
                            textInput('qp_Cvector', 'C', placeholder = '[1,0,1]'),
                            textInput('qp_X0', 'X0', placeholder = '[3,5,7]'),
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
                            tabsetPanel(
                                 tabPanel("Table", DT::dataTableOutput('qp_table')),
                                 tabPanel("Plot", plotOutput('qp_graph'))
                            )
                        )
                    )
            ),
            
            tabItem('gd_rosenbrock',
                    h1('GD Funcion de Rosenbrock'),
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            textInput('rf_X0', 'X0', placeholder = '[0, 0]'),
                            numericInput('rf_epsilon', 'Tolerancia (e)', value = 0.00000001),
                            numericInput('rf_iters', 'N', value = 1000),
                            numericInput('rf_lr', 'Learning Rate', value = 0.05),
                            actionButton('eval_rf', 'Evaluar GD')
                        ),
                        box(width = 8, solidHeader = TRUE,
                            DT::dataTableOutput('rf_table')
                        )
                    )
            ),
            
            
            tabItem('generar_datos',
                    h1('Generar Datos'),
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            numericInput('data_d', 'Columnas (d)', value = 100),
                            numericInput('data_n', 'Observaciones (n)', value = 1000),
                            textInput('data_path', 'Folder path', placeholder = 'C:/user/folder/datos.pickle'),
                            actionButton('data_run', 'Generar datos')
                        ),
                        box(width = 8, solidHeader = TRUE,
                            textOutput('save_data')
                        )
                    )
            ),
            
            
            tabItem('gd_solver',
                    h1('GD Variants Solver'), 
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            radioButtons('gds_algorithm',
                                         'Seleccione una variante de GD',
                                         choices = c('Solucion Cerrada' = 'closeGD',
                                                     'GD' = 'GD',
                                                     'SGD' = 'SGD',
                                                     'MBGD' = 'MBGD'),
                                         selected = 'closeGD'),
                            textInput('gds_path', 'File path', placeholder = 'C:/user/folder/datos.pickle'),
                            numericInput('gds_epsilon', 'Tolerancia (e)', value = 0.00000001),
                            numericInput('gds_lr', 'Learning Rate', value = 0.0005),
                            numericInput('gds_iters', 'Kmax', value = 1000),
                            numericInput('gds_mb', 'Mini-batch size', value = 25),
                            p('*Minibatch solo sera utilizado para MB-GD'),
                            actionButton('eval_gds', 'Evaluar GD')
                        ),
                        box(width = 8, solidHeader = TRUE,
                            DT::dataTableOutput('gds_table')
                        )
                    )
            ),
            
            
            tabItem('bt_solver',
                    h1('Rosenbrock con Backtracking'), 
                    fluidRow(
                        box(width = 4, solidHeader = TRUE, status = "primary",
                            radioButtons('bt_algorithm',
                                         'Seleccione algoritmo',
                                         choices = c('GD' = 'GD',
                                                     'Metodo Newton' = 'Newton'),
                                         selected = 'GD'),
                            radioButtons('bt_lr_rb',
                                         'Seleccione tipo LR',
                                         choices = c('backtracking' = 'backtracking',
                                                     'constante' = 'constant'),
                                         selected = 'backtracking'),
                            textInput('bt_X0', 'X0', placeholder = '[0, 0]'),
                            numericInput('bt_epsilon', 'Tolerancia (e)', value = 0.00000001),
                            numericInput('bt_lr', 'Learning Rate', value = 1),
                            numericInput('bt_iters', 'Kmax', value = 1000),
                            actionButton('eval_bt', 'Evaluar Backtracking')
                        ),
                        box(width = 8, solidHeader = TRUE,
                            DT::dataTableOutput('bt_table')
                        )
                    )
            )
            
            
        )
    )
)
