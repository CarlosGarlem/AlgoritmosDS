library(shiny)
library(reticulate)
use_python('D:/anaconda3/envs/pydsEnv')

source_python("algoritmos.py")


shinyServer(function(input, output) {
    
    #REACTIVE BUTTONS -----------------------------------
    
    algorithm_option <- reactive({
        input$algorithms_diffnum
    })
    
    algorithm_option_r2 <- reactive({
        input$algorithms_diffnum_r2
    })
    
    #REACTIVE FUNCTIONS -----------------------------------
    
    calculateDiffNum <- eventReactive(input$eval_diffnum, {
        equation <- input$function_diffnum[1]
        x <- input$x_diffnum[1]
        h <- input$h_diffnum[1]
        
        d_function <- switch(algorithm_option(),
                             diffCenter = center_finite_derivative,
                             diffProgressive = progressive_finite_derivative,
                             diffCenter2 = center_finite_derivative_2)
        derivative <- d_function(equation, x, h)
        
        df <- data.frame('Variable' = c('x'), 'Ecuacion' = equation, 'h' = h, 'Aproximacion' = derivative)
        df
        
    })
    
    
    calculateDiffNum_r2 <- eventReactive(input$eval_diffnum_r2, {
        equation <- input$function_diffnum_r2[1]
        x <- input$x_diffnum_r2[1]
        x <- gsub('[() ]', '', x)
        x <- strsplit(x, ',')
        x <- unlist(x, recursive = TRUE)
        h <- input$h_diffnum_r2[1]
        
        
        d_function <- switch(algorithm_option_r2(),
                             diffCenter = center_finite_derivative_r2,
                             diffProgressive = progressive_finite_derivative_r2,
                             diffCenter2 = center_finite_derivative_2_r2)
        derivative <- d_function(equation, x, h)
        
        df <- data.frame('Variable' = c('x', 'y'), 'Ecuacion' = equation, 'h' = h, 'Aproximacion' = derivative)
        df
    })

    
    calculateNewton <- eventReactive(input$eval_newton, {
        equation <- input$function_newton[1]
        x <- input$newton_x0[1]
        e <- input$newton_e[1]
        k <- input$newton_iters[1]
        
        result <- metodo_newton(equation, x, k, e)
        df <- as.data.frame(result)
        df
        
    })
    
    
    calculateBiseccion <- eventReactive(input$eval_biseccion, {
        equation <- input$function_biseccion[1]
        e <- input$biseccion_e[1]
        k <- input$biseccion_iters[1]
        
        dominio <- input$biseccion_interval[1]
        dominio <- gsub('(\\[|\\]| )', '', dominio)
        dominio <- strsplit(dominio, ',')
        dominio <- unlist(dominio, recursive = TRUE)
        
        result <- metodo_biseccion(equation, dominio, k, e)
        df <- as.data.frame(result)
        df
        
    })
    
    #RENDER OUTPUTS  --------------------------------------------------
    #output$diffnum_table <- DT::renderDataTable({
    #    calculateDiffNum()
    #})
    #Derivadas
    output$diffnum_table <- renderTable({
        calculateDiffNum()
    })
    
    output$diffnum_table_r2 <- renderTable({
        calculateDiffNum_r2()
    })
    
    #Ceros
    output$newton_table<- DT::renderDataTable({
        calculateNewton()
    })
    
    
    output$biseccion_table<- DT::renderDataTable({
        calculateBiseccion()
    })
    
    
})
