library(shiny)
library(reticulate)
library(ggplot2)
use_python('D:/anaconda3/envs/pydsEnv')
source_python("algoritmos.py")


shinyServer(function(input, output) {
    

    # Auxiliary functions ---------------------------------
    parse_input <- function(input, type){
        x <- input
        x <- gsub('(\\],)', ']|', x)
        x <- gsub('(\\[|\\])', '', x)
        
        if(type == 'matrix'){
            x <- strsplit(x, '|', fixed = T)
            x <- unlist(x, recursive = T)
            x <- strsplit(x, ',')
        } else {
            x <- strsplit(x, ',')
            x <- unlist(x, recursive = TRUE)
        }
        
        return(x)
    }
    
    #REACTIVE BUTTONS -------------------------------------
    
    algorithm_option <- reactive({
        input$algorithms_diffnum
    })
    
    algorithm_option_r2 <- reactive({
        input$algorithms_diffnum_r2
    })
    
    qp_lrType <- reactive({
        input$qp_rbuttons
    })
    
    gds_algorithm_option <- reactive({
        input$gds_algorithm
    })
    
    bt_algorithm_option <- reactive({
        input$bt_algorithm
    })
    
    bt_lr_option <- reactive({
        input$bt_lr_rb
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
        
        df <- metodo_newton(equation, x, k, e)
        #write.csv(df, './newton.csv')
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
        
        df <- metodo_biseccion(equation, dominio, k, e)
        #write.csv(df, './biseccion.csv')
        df
        
    })
    
   
    
    calculateGDQP <- eventReactive(input$eval_qp, {
        Q <- parse_input(input$qp_Qmatrix[1], 'matrix')
        c <- parse_input(input$qp_Cvector[1], 'vector')
        x <- parse_input(input$qp_X0[1], 'vector')
        e <- input$qp_epsilon[1]
        N <- input$qp_iters[1]
        lr <- input$qp_lr[1]
        lr_type <- qp_lrType()
        
        #output <- parseInput(Q, F)
        #as.character(output)
        df <- gradient_descent_QP(x, Q, c, N, e, lr_type, lr)
        df$Xn <- as.character(df$Xn)
        df$Pk <- as.character(df$Pk)
        #write.csv(df, paste('QP', '_', as.character(lr_type), '_', as.character(lr), '.csv', sep = ''))
        df
        
    })
    
    
    calculateGDRosenbrock <- eventReactive(input$eval_rf, {
        x <- parse_input(input$rf_X0[1], 'vector')
        e <- input$rf_epsilon[1]
        N <- input$rf_iters[1]
        lr <- input$rf_lr[1]
        
        df <- rosenbrock_gd(x, N, e, lr)
        df$Xn <- as.character(df$Xn)
        df$Pk <- as.character(df$Pk)
        #write.csv(df, './rf.csv')
        df
        
    })
    
    
    generateMatrixData <- eventReactive(input$data_run, {
        d <- input$data_d[1]
        n <- input$data_n[1]
        path <- input$data_path[1]
        
        txt <- generateData(d, n, path)
        txt
    })
    
    
    calculateGDVariant <- eventReactive(input$eval_gds, {
        path <- input$gds_path[1]
        e <- input$gds_epsilon[1]
        kmax <- input$gds_iters[1]
        lr <- input$gds_lr[1]
        mb <- input$gds_mb[1]
        variant <- gds_algorithm_option()

        if(variant == 'closeGD'){
            df <- getCloseSolution(path)
        }else{
            df <- gdSolver(path, kmax, lr, mb, variant, e)
            df$Pk <- as.character(df$Pk)
            df$Pk <- substr(df$Pk, 1, 50)
            df$Pk <- paste(df$Pk, '...', sep = '')
        }
        df$Xn <- as.character(df$Xn)
        df$Xn <- substr(df$Xn, 1, 50)
        df$Xn <- paste(df$Xn, '...', sep = '')
        df
    })
    
    
    calculateBTRosenbrock <- eventReactive(input$eval_bt, {
        x <- parse_input(input$bt_X0[1], 'vector')
        e <- input$bt_epsilon[1]
        kmax <- input$bt_iters[1]
        lr <- input$bt_lr[1]
        lr_type <- bt_lr_option()
        
        d_function <- switch(bt_algorithm_option(),
                             GD = rosenbrock_backtracking,
                             Newton = newton_optimization)
        
        df <- d_function(x, kmax, e, lr, lr_type)
        df$Xn <- as.character(df$Xn)
        df$Pk <- as.character(df$Pk)
        df
        
    })
    
    
    #RENDER OUTPUTS  --------------------------------------------------

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
    
    
    #Gradient Descent
    output$qp_table<- DT::renderDataTable({
        calculateGDQP()
    })
    
    
    output$qp_graph<- renderPlot({
        df <- calculateGDQP()
        ggplot(df, aes(x = Iter, y = Error)) +
             geom_point(color = 'blue') + 
             geom_line(color = 'blue') +
             ggtitle('Grafica de Errores') + 
             theme_minimal()
    })
    
    
    output$rf_table<- DT::renderDataTable({
        calculateGDRosenbrock()
    })
    
    
    #Salvar datos
    output$save_data <- renderText({
        generateMatrixData()
    })
    
    
    #GD variants output
    output$gds_table<- DT::renderDataTable({
        calculateGDVariant()
    })
    
    
    output$bt_table<- DT::renderDataTable({
        calculateBTRosenbrock()
    })
    
})
