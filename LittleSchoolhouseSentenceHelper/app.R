#load packages
library(shiny)
library(shinythemes)
library(dplyr)
library(readr)

#Create Python virtual environment and install transformers package
reticulate::virtualenv_create("hugface9", packages=c("transformers[tf-cpu]"))
reticulate::use_virtualenv("hugface9", required = TRUE)

#Import tokenizer and model from transformers
transformers <- reticulate::import("transformers", delay_load=TRUE)
GPT2Tokenizer <- transformers$GPT2Tokenizer
TFGPT2LMHeadModel <- transformers$TFGPT2LMHeadModel

#Load data - Using on-line models
tokenizer <- GPT2Tokenizer$from_pretrained("gpt2")
model <- TFGPT2LMHeadModel$from_pretrained("gpt2")
#Set maximum starter phrase length
model$config$max_length = as.integer(50)

#Define UI
ui <- fluidPage(
    titlePanel("The Little Schoolhouse Sentence Helper"),
    textInput(inputId = "prompt",
              label = "Enter a starting phrase"),
    submitButton(text = "Submit", icon = NULL, width = NULL),
    textOutput(outputId = "generated")
)

#Define server function
generate_text <- function(prompt){
    inputs <- tokenizer$encode(prompt, return_tensors="tf")
    outputs <- model$generate(inputs, do_sample=TRUE)
    generated = tokenizer$decode(outputs[0])
}

#Use validate and need to hold off running generate_text until prompt has a value
server <- function(input, output) {
    output$generated <- renderText({
        validate(
            need(input$prompt, "Please enter some words")
        )
        generated <- generate_text(input$prompt)
        print(generated)
    })
}

# Create Shiny object
shinyApp(ui = ui, server = server)
