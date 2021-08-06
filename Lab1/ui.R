library(shiny)
library(shinydashboard)
#library(DT)

# Define UI for application that draws a histogram
dashboardPage(
    
    dashboardHeader(title = 'Algoritmos en la Ciencia de Datos'),
    
    dashboardSidebar(
        sidebarMenu(
            menuItem('Ceros', tabName = 'Ceros'),
            menuItem('Derivacion', tabName = 'Derivacion'),
            menuItem('Diferenciacion Numerica', tabName = 'diffNum')
        )
    ),
    
    dashboardBody(
        tabItems(
            tabItem('Ceros',
                    h1('Metodo de Newton'),
                    box(textInput('ecuacion', 'Ingrese la Ecuacion'),
                        textInput('initVal', 'X0'),
                        textInput('Error', 'Error'),
                        actionButton('nwtSolver', 'Newton Solver')
                    ),
                    tableOutput('salidaTabla')
            ),
            
            tabItem('Derivacion',
                    h1('Diferencias Finitas'),
                    box(textInput('difFinEcu', 'Ingrese la Ecuacion'),
                        textInput('valorX', 'x'),
                        textInput('valorH', 'h'),
                        actionButton('diferFinEval', 'Evaluar Derivada')
                    ),
                    textOutput('difFinitOut')
            ),
            
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
                            numericInput('x_diffnum', 'x', value = 1),
                            numericInput('h_diffnum', 'h', value = 1),
                            actionButton('eval_diffnum', 'Evaluar Derivada')
                        ),
                        box(width = 4, solidHeader = TRUE,
                            #DT::dataTableOutput('diffnum_table')
                            tableOutput('diffnum_table')
                        )
                    )
            )
        )
    )
)
