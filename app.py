import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to generate a blog response using the LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    """
    This function generates a blog response using the LLaMA 2 model.
    
    Parameters:
    - input_text (str): The topic of the blog.
    - no_words (int): The desired word count for the blog.
    - blog_style (str): The target audience for the blog (e.g., Researchers, Data Scientists, Common People).
    
    Returns:
    - response (str): The generated blog content.
    """
    
    ### Load the LLaMA 2 model using CTransformers
    llm = CTransformers(
        model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Path to the LLaMA 2 model file
        model_type='llama',  # Specify the model type
        config={
            'max_new_tokens': 256,  # Limit the generated response to 256 tokens
            'temperature': 0.01  # Control randomness (lower means more deterministic output)
        }
    )
    
    ## Define a prompt template to structure the input for the model
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    # Create a PromptTemplate object using the defined template
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )
    
    ## Generate the response from the LLaMA 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    
    print(response)  # Print the response in the console for debugging
    return response  # Return the generated response


## Streamlit UI configuration
st.set_page_config(
    page_title="Generate Blogs",  # Title of the web app
    page_icon='🤖',  # Favicon (emoji) for the app
    layout='centered',  # Layout style
    initial_sidebar_state='collapsed'  # Hide sidebar by default
)

# Display the app title
st.header("Generate Blogs 🤖")

# Input field for entering the blog topic
input_text = st.text_input("Enter the Blog Topic")

## Creating two columns for additional input fields
col1, col2 = st.columns([5, 5])

with col1:
    # Input field for specifying the desired word count
    no_words = st.text_input('No of Words')

with col2:
    # Dropdown to select the target audience for the blog
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0  # Default selection
    )

# Button to trigger blog generation
submit = st.button("Generate")

## If the button is clicked, generate and display the blog response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
