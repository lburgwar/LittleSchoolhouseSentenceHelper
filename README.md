# LittleSchoolhouseSentenceHelper
The Little Schoolhouse Sentence Helper puts a twist on an age old technique that language educators have used. In elementary school students are given work sheets that present a phrase followed by a blank line. The student is to then complete the sentence ensuring it makes sense and that the the result is a complete sentence. With the Little Schoolhouse Sentence Helper the student is allowed to create their own initial phrase and then see how the phrase is completed forming a complete sentence. This teaches the student what a complete sentence should look like and provides a model for the student to follow.
Using the Little Schoolhouse Sentence Helper is very easy. Simply start the helper by entering the URL in a browser:
 https://lburgwar.shinyapps.io/shinyTextGenerator4/
Next, enter a starting phrase of between one and fifty words. Then click on the submit button and wait for the Little Schoolhouse Sentence Helper to gerate the follow-on text.

The approach we used to build the Little Schoolhouse Sentence Helper is based on the OpenAI GPT-2 model. GPT-2 is a language model trained on eight million web pages. This provides a natural flow of words following a starting phrase. GPT-2 is an open source pre-trained model based on transformer networks. Transformers is a ground breaking technology introduced to the NLP community in [1]. Transformers is the latest step in the transition from RNN's, GRU's, and LSTM. Transformers is more complex than the previous architectures but it provides a much richer understanding of the text being worked than the other three. Transformers uses a "self attention" technique which takes each word and studies how a question about that word can be answered  (or at least hinted at) by each of the other words. Questions such as what?, who? and when? are among the eight or so questions asked of each word. Self attention is useful in language translation, question answering and sentiment analysis. In the context of the Little Schoolhouse Sentence Helper the rich understanding of how each word fits with other words in the the starting phrase along with the words in the training corpus leads to a strong coherent generation of text. 

To apply transformer technology in an R based Shiny app we used a bridge between the R language and Python. The "reticulate" module in R provides that bridge and enables the use of Python, a very important data science language, into R. In the code below you can see how reticulate is used to import the transformers package. In this case we selected the tensorflow (tf) version to run on a cpu rather than gpu. In Python importing modules using the "from X import Y" syntax is often used to limit the amount of code brought into memory. Here we used the delay_load parameter and extracted the GPT2Tokenizer and TFGPT2LMHeadModel into R. Finally the pretrained model components for the tokenizer and the model itself are brought into R.
```{r, eval=FALSE}
#load packages
library(shiny)
library(shinythemes)
library(dplyr)
library(readr)

#Create virtual environment 
reticulate::virtualenv_create("hugface9", packages=c("transformers[tf-cpu]"))
reticulate::use_virtualenv("hugface9", required = TRUE)

#Import the transformer components into R
transformers <- reticulate::import("transformers", delay_load=TRUE)
GPT2Tokenizer <- transformers$GPT2Tokenizer
TFGPT2LMHeadModel <- transformers$TFGPT2LMHeadModel

#Load data - Using on-line model GPT-2
tokenizer <- GPT2Tokenizer$from_pretrained("gpt2")
model <- TFGPT2LMHeadModel$from_pretrained("gpt2")
#Set limit on starting phrase length in words
model$config$max_length = as.integer(50)
```

The ui and server parts of the Shiny app are shwon below. This is a straight forward user interface with an input box and printout of the results. The server function "generate_text" takes in the starting phrase labeled as "prompt" and encodes it using the transformer tokenizer. This basically looks up each word in the starting phrase in the vocabulary and assigns it a number. The server function itself takes the list of encoded words and presents it to the renderText function. The result is then printed for display. 

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

#Use validate and need to hold off running generate_text until the prompt has a value
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


References
[1] 
