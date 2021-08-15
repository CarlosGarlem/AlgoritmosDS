library(shiny)
library(shinydashboard)
#library(DT)

# Define UI for application that draws a histogram
dashboardPage(
    
    dashboardHeader(title = 'Algoritmos DS'),
    
    dashboardSidebar(
        sidebarMenu(
            menuItem('Diferenciacion Numerica en R', tabName = 'diffNum'),
            menuItem('Diferenciacion Numerica en R2', tabName = 'diffNumR2'),
            menuItem('Metodo Newton', tabName = 'ceros_newton'),
            menuItem('Metodo Biseccion', tabName = 'ceros_biseccion')
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
                            #DT::dataTableOutput('diffnum_table')
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
                            #DT::dataTableOutput('diffnum_table')
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
                            #tableOutput('newton_table')
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
                            #tableOutput('biseccion_table')
                            DT::dataTableOutput('biseccion_table')
                        )
                    )
            )
        )
    )
)
