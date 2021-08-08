library(shiny)
library(reticulate)
use_python('D:/anaconda3/envs/pydsEnv')

source_python("algoritmos.py")


shinyServer(function(input, output) {
    
    #Evento y evaluacion de metodo de newton para ceros
    newtonCalculate<-eventReactive(input$nwtSolver, {
        inputEcStr<-input$ecuacion[1]
        print(inputEcStr)
        initVal<-input$initVal[1]
        error<-input$Error[1]
        #outs<-add(initVal, error)
        outs<-newtonSolverX(initVal, inputEcStr, error)
        outs
    })
    
    #Evento y evaluación de diferencias finitas
    diferFinitCalculate<-eventReactive(input$diferFinEval, {
        inputEcStr<-input$difFinEcu[1]
        valX<-input$valorX[1]
        h<-input$valorH[1]
        outs<-evaluate_derivate_fx(inputEcStr, valX, h)
        as.character(outs)
    })
    
    
    algorithm_option <- reactive({
        input$algorithms_diffnum
    })
    
    algorithm_option_r2 <- reactive({
        input$algorithms_diffnum_r2
    })
    
    
    calculateDiffNum <- eventReactive(input$eval_diffnum, {
        equation <- input$function_diffnum[1]
        x <- input$x_diffnum[1]
        h <- input$h_diffnum[1]
        
        d_function <- switch(algorithm_option(),
                             diffCenter = center_finite_derivative,
                             diffProgressive = progressive_finite_derivative,
                             diffCenter2 = center_finite_derivative_2)
        derivative <- d_function(equation, x, h)

        
        d <- '12x^3 - 6x^2 - 8x +5'
        d <- gsub('x', paste('*(', x, ')', sep = ''), d)
        d_value <- eval(parse(text = d))
        error <- abs(d_value - derivative)
        
        df <- data.frame('Variable' = c('x'), 'Valor Real' = d_value, 'Aproximacion' = derivative, 'Error' = error)
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
        
        

        d1 <- '12x^3 - 6x^2y - 8xy^2 + 5y^3'
        d1 <- gsub('x', paste('*(', x[1], ')', sep = ''), d1)
        d1 <- gsub('y', paste('*(', x[2], ')', sep = ''), d1)
        
        d2 <- '-2x^3 - 8x^2y + 15xy^2 + 8y^3'
        d2 <- gsub('x', paste('*(', x[1], ')', sep = ''), d2)
        d2 <- gsub('y', paste('*(', x[2], ')', sep = ''), d2)
        
        
        d_value <- c(eval(parse(text = d1))
                     ,eval(parse(text = d2)))
        
        error <- abs(d_value - derivative)
        df <- data.frame('Variable' = c('x', 'y'), 'Valor Real' = d_value, 'Aproximacion' = derivative, 'Error' = error)
        df
        
    })

    
    #REnder metodo de Newton
    output$salidaTabla<-renderTable({
        newtonCalculate()
    })
    
    
    #Render Diferncias Finitas
    output$difFinitOut<-renderText({
        diferFinitCalculate()
    })
    
    
    #output$diffnum_table <- DT::renderDataTable({
    #    calculateDiffNum()
    #})
    output$diffnum_table <- renderTable({
        calculateDiffNum()
    })
    
    output$diffnum_table_r2 <- renderTable({
        calculateDiffNum_r2()
    })
    
})
