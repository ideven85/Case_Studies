# ShopAssist AI


## Part 1: Introduction




#### Project Background

In today's digital age, online shopping has become the go-to option for many consumers. However, the overwhelming number of choices and the lack of personalized assistance can make the shopping experience daunting. To address this, we have developed **ShopAssist AI, a chatbot that combines the power of large language models and rule-based functions to ensure accurate and reliable information delivery**.


#### Problem Statement

*Given a dataset containing information about laptops (product names, specifications, descriptions, etc.), build a chatbot that parses the dataset and provides accurate laptop recommendations based on user requirements*.


You can load the data and see it here.


```python
# Install OpenAI library
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/387.1 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[90m╺[0m [32m378.9/387.1 kB[0m [31m12.5 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m387.1/387.1 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python

```

    Mounted at /content/drive



```python
# Import other libraries
import pandas as pd
from IPython.display import display, HTML

pd.set_option("display.width", 100)
```


```python
filepath = "/content/drive/MyDrive/GenAI/"
```

#### Approach:

1. **Conversation and Information Gathering**: The chatbot will utilize language models to understand and generate natural responses. Through a conversational flow, it will ask relevant questions to gather information about the user's requirements.
2. **Information Extraction**: Once the essential information is collected, rule-based functions come into play, extracting top 3 laptops that best matches the user's needs.
3. **Personalized Recommendation**: Leveraging this extracted information, the chatbot engages in further dialogue with the user, efficiently addressing their queries and aiding them in finding the perfect laptop solution.

## Part 2: System Design


#### Dataset

We have a dataset `laptop.csv` where  each row describes the features of a single laptop and also has a small description at the end. The chatbot that we build will leverage LLMs to parse this `Description` column and provide recommendations

#### Workings of the Chatbot



The chatbot should ask a series of questions to
- Determine the user's requirments. For simplicity, we have used 6 features to encapsulate the user's needs. The 6 features are as follows:
    - GPU intensity
    - Display quality
    - Portability
    - Multitasking
    - Processing speed
    - Budget

- Confirm if the user's requirements have been correctly captured at the end.

After that the chatbot lists down the top 3 products that are the most relevant, and engages in further conversation to help the user find the best one.


# ShopAssist 1.0

#### Building the Chatbot

Below is the system design for the chatbot for **ShopAssist 1.0**.

![Chatbot_sys_design.png](https://drive.google.com/uc?id=1j-mw_dNcbxGcelQ0PmkDB0nKOpauU1wX)

|`Stage 1`

- Intent Clarity Layer
- Intent Confirmation Layer

`Stage 2`

- Product Mapping Layer
- Product Information Extraction Layer

`Stage 3`

- Product Recommendation Layer

##### Major functions behind the Chatbot

Let's now look at a brief overview of the major functions that form the chatbot. We'll take a deep dive later



- `initialize_conversation()`: This initializes the variable conversation with the system message.
- `get_chat_completions()`: This takes the ongoing conversation as the input and returns the response by the assistant
- `moderation_check()`: This checks if the user's or the assistant's message is inappropriate. If any of these is inappropriate, it ends the conversation.
- `intent_confirmation_layer()`: This function takes the assistant's response and evaluates if the chatbot has captured the user's profile clearly. Specifically, this checks if the following properties for the user has been captured or not GPU intensity, Display quality, Portability, Multitasking, Processing speed, Budget
- `dictionary_present()`: This function checks if the final understanding of user's profile is returned by the chatbot as a python dictionary or not. If there is a dictionary, it extracts the information as a Python dictionary.
- `compare_laptops_with_user()`: This function compares the user's profile with the different laptops and come back with the top 3 recommendations.
- `initialize_conv_reco()`: Initializes the recommendations conversation

In the next sections, we will look at how to write the code for the above functions.

# ShopAssist 2.0

But in development of **ShopAssist2.0** , we leverage from Function Calling feature for improved performance and simple architecture.

The architecture looks as shown below:

|`Stage 1`

- Intent Clarity Layer

`Stage 2`

- Product Mapping Layer

`Stage 3`

- Product Recommendation Layer

The following layers from ShopAssist 1.0 **are removed** as their purpose are filled by now using function calling in OpenAI API.

1. intent_confirmation_layer()
2. dictionary_present()
3. initialize_conv_reco()

## Part 3: Implementation


```python

```

## Stage 1

### 3.1 - Import the libraries

Let's start by importing the libraries that we'll require for this project. Following are the ones:
- openai
- pandas
- os, json, ast

Make sure the api key is stored in the text file `OPENAI_API_Key.txt`.


```python
# Import the libraries
import os, json, ast
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
```


```python
# Read the OpenAI API key
with open(filepath + "OpenAI_API_Key.txt", "r") as f:
    openai.api_key = " ".join(f.readlines())
```

### 3.2 - Implementing Intent Clarity Layer


- `initialize_conversation()`


### `initialize_conversation()`:
This initializes the variable conversation with the system message. Using prompt engineering and chain of thought reasoning, the function will enable the chatbot to keep asking questions until the user requirements have been captured in a dictionary. It also includes Few Shot Prompting(sample conversation between the user and assistant) to align the model about user and assistant responses at each step.


```python
# initialize_conversation function
def initialize_conversation():
    """
    Returns a list [{'role':'system','content': system_message}]
    """
    delimiter = "####"

    system_message = f"""
  You are very intelligent laptop gadget expert and your goal is to find out best laptop for a user for given input.
  You need to ask relevant questions to the user and understand the user profile  by analysing the user's responses.
  Your final objective is to get following parameters from user's conversation. The parameters are 'GPU intensity','Display quality','Portability','Multitasking','Processing speed'and'Budget', and be confident of the values.
  You need to ask probing questions to the user in case any of these parameters are not obtained from user's interaction.
  The value for the 'Budget' should be numerical value extracted from user's response.
  The value of all keys except 'Budget' should be either 'high' or 'medium' or 'low' based on the importance of the corresponding keys as stated by user.
  All the values in the example dictionary are only representative values.
  {delimiter}
  Here are some instructions around the values for the different keys. if you do not follow this, you will be heavily penalized.
  - The value of all keys except 'Budget' should be strictly either 'high' or 'medium' or 'low' based on the importance of the corresponding keys as stated by user.
  - The value for the 'Budget' should be numerical value extracted from user's response.
  - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention to user that there are no laptops in that range.
  - Please do not randomly assign any value to the parameters.
  - The value needs to be inferred from the user's response.
  - Please ask one question at a time to capture values for the parameters.
  {delimiter}

  Once you have obtained all the parameter, your goal is to extract details of the top 3 laptops matching to the parameters obtaned earlier in the user's conversation .
  And this you will do by function calling with compare_laptops_with_user function.

  {delimiter}
    Once you get the list of top 3 laptops you will need to present the details in the personlized format it and show the recomendations to the
    user in the following format:
    1. <Laptop name>: <Basic laptop specs in brief>, <price of laptop>
    2. <Laptop name>: <Basic laptop specs in brief>, <price of laptop>
    3. <Laptop name>: <Basic laptop specs in brief>, <price of laptop>
  {delimiter}

  Follow below steps when interacting with user.

  Step 1 : Start with a short welcome message and encourage the user to share their requirement precisely on the laptop needed.
  Remember, you only recommend on laptop, if user is asking anything else then you apologizes and remind user that you are laptop expert only. Please do not ask more than one questions at a time.

  Step 2 : Based on the user input obtained from conversation, you try to get values for 'GPU intensity','Display quality','Portability','Multitasking','Processing speed'and'Budget'.

  Step 3 : In case, details are not clear , ask clarifying question to user to get the details for above parameters.

  Step 4 : When all the parameters are available you invoke function calling with compare_laptops_with_user function.

  Step 5 : Personalize the recommendation obtained from compare_laptops_with_user function and present to the user.

  Step 6 : If user changes any of the request then repeat step 2 to 5.


  {delimiter}
  Here are some notes which you should take care of. If you do not folow this you will be heavily penalized.
  - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
  - The value for 'budget' should be a numerical value extracted from the user's response. It needs to be greater than or equal to 25000 INR.
  - Do not ask questions to user for more 1 parameter at a time.
  - Do not ask too many questions in one shot.
  - Do not reply on any other subject other than laptop and its parameters.


    {delimiter}
  Her is a sample conversation between user and AI assistant:
  User : 'Hi, I am an editor.'
  Assistant : 'Great! As an editor, you likely require a laptop tha can handle demanding tasks. hence , the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on ? Are you more involved in video editing , photo editing or both ? Understanding the specific type of editing work will help me to tailor my recommendation accordingly. Let me know if my understanding is correct until now.  '
  User : ' I primarily work with after effects. '
  Assistant : ' Thank you for providing that information. Working with after effects involves working with graphics, animations and rendering which will require high GPU. Do you work with high resolution media files, such as 4K videos or RAW photos ? Understanding your file sizes will help determine the stoprage capacity and processing power needed. '
  User : ' Yes, sometimes I work with 4K videos as well. '
  Assistant : ' Thank you for the information. Processing 4K videos will require a good and high GPU. I think we have already determined earlier that you need a high CPU. To ensure that I have a complete understanding of your needs, I have one more question : Are you frequently on the go and require a laptop that s light weight and easy to carry, or do you primarily work from a stationary location ?  '
  User : ' Yes, sometimes I travel but do not carry my laptop. '
  Assistant : ' Could you kindly let me know your budget for the laptop ? This will help me to find the options that fit within your price range while meeting the specified requirements. '
  User : ' My maximum budget is 1.5 lakh INR. '

  {delimiter}

  {delimiter}
  Here is another sample conversation between the user and assistant:
  User: "I am a coder and i need higher end laptop with high multitasking feature"
  Assistant: "Great! As a coder, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. Could you please specify the importance of GPU intensity, display quality, portability, processing speed, and budget in order to tailor my recommendation accordingly?"
  User: "Yes, you are right"
  Assistant: "Great! Do you prefer a laptop that is easy to carry around or do you primarily use it at a fixed location?"
  User: "I only need it for home use."
  Assistant: "Thank you for specifying that. Do you often have multiple tasks running on your laptop simultaneously or primarily focus on one task at a time?"
  User: "I ususally have multiple tasks running"
  Assistant: "Thank you for sharing that. Do you prefer a laptop with fast processing speed for quick performance?"
  User: "yes i need it to be very quick"
  Assistant: "Appreciate that input. Finally, what is your budget range for the laptop you are looking to purchase?"
  User: "i would like if it is not more than 125000 INR"


  {delimiter}
  Here is another sample conversation between the user and assistant:
  User: "I am a basic business user"
  Assistant: "As a basic business user, it's important to have a reliable laptop for your day-to-day tasks. Based on your usage as a basic business user, do you prefer a laptop with a good quality display for presentations?"
  User: "Yes, you right"
  Assistant: "Great! Do you prefer a laptop that is easy to carry around or do you primarily use it at a fixed location?"
  User: "I need it to be easy to carry around as I travel frequently on business trips"
  Assistant: "Thank you for specifying that. Do you often have multiple tasks running on your laptop simultaneously or primarily focus on one task at a time?"
  User: "i ususally have multiple tasks running"
  Assistant: "Thank you for sharing that. Do you prefer a laptop with fast processing speed for quick performance?"
  User: "yes i need it to be very quick"
  Assistant: "Appreciate that input. Finally, what is your budget range for the laptop you are looking to purchase?"
  User: "i would like if it is not more than 1 lakh"


  """

    conversation = [{"role": "system", "content": system_message}]

    return conversation
```

Let's now look at the next function.
- `get_chat_completions()`: This takes the ongoing conversation as the input and returns the response by the assistant. We'll use the Chat Completions function for performing LLM calls to OpenAI.

### `get_chat_completions()`:

This function perform LLM call using the Chat Completions API to get the LLM response.


```python
# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(inp, json_format=False):

    MODEL = "gpt-4o-mini"

    system_message_json_output = (
        """<<. Return output in JSON format to the key output.>>"""
    )

    try:
        # If the output is required to be in JSON format
        if json_format:
            # Append the input prompt to include JSON response as specified by OpenAI
            messages = inp + system_message_json_output

            # JSON return type specified
            chat_completion_json = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                seed=1234,
            )

            output = json.loads(chat_completion_json.choices[0].message.content)

        # No JSON return type specified
        else:
            message = inp
            chat_completion = openai.chat.completions.create(
                model=MODEL, messages=message, seed=2345
            )

            output = chat_completion.choices[0].message.content
        return output

    # Handling exception
    except Exception as e:
        print(f"An error occurred while calling OpenAI API :  {e}")
        return "None"
```

### `moderation_check()`:
 This checks if the user's or the assistant's message is inappropriate. If any of these is inappropriate, you can add a break statement to end the conversation.


```python
# Define a function called moderation_check that takes user_input as a parameter.


def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    response = openai.moderations.create(input=user_input)

    # Extract the moderation result from the API response.
    moderation_output = response.results[0].flagged
    # Check if the input was flagged by the moderation system.
    if response.results[0].flagged == True:
        # If flagged, return "Flagged"
        return "Flagged"
    else:
        # If not flagged, return "Not Flagged"
        return "Not Flagged"
```

## Stage 2

### 3.3 Implementing the Product Mapping


### `product_map_layer()`:

This function is responsible for extracting key features and criteria from laptop descriptions. Here's a breakdown of how it works:

-  Use a prompt that assign it the role of a Laptop Specifications Classifier, whose objective is to extract key features and classify them based on laptop descriptions.

- Provide step-by-step instructions for extracting laptop features from description.

- Assign specific rules for each feature (e.g., GPU Intensity, Display Quality, Portability, Multitasking, Processing Speed) and associate them with the appropriate classification value (Low, Medium, or High).

- Includes Few Shot Prompting (sample conversation between the user and assistant) to demonstrate the expected result of the feature extraction and classification process.


```python
def product_map_layer(laptop_description):
    delimiter = "#####"

    lap_spec = {
        "GPU intensity": "(Type of the Graphics Processor)",
        "Display quality": "(Display Type, Screen Resolution, Display Size)",
        "Portability": "(Laptop Weight)",
        "Multitasking": "(RAM Size)",
        "Processing speed": "(CPU Type, Core, Clock Speed)",
    }

    values = {"low", "medium", "high"}

    prompt = f"""
    You are a Laptop Specifications Classifier whose job is to extract the key features of laptops and classify them as per their requirements.
    To analyze each laptop, perform the following steps:
    Step 1: Extract the laptop's primary features from the description {laptop_description}
    Step 2: Store the extracted features in {lap_spec} \
    Step 3: Classify each of the items in {lap_spec} into {values} based on the following rules: \
    {delimiter}
    GPU Intensity:
    - low: <<< if GPU is entry-level such as an integrated graphics processor or entry-level dedicated graphics like Intel UHD >>> , \n
    - medium: <<< if mid-range dedicated graphics like M1, AMD Radeon, Intel Iris >>> , \n
    - high: <<< high-end dedicated graphics like Nvidia RTX >>> , \n

    Display Quality:
    - low: <<< if resolution is below Full HD (e.g., 1366x768). >>> , \n
    - medium: <<< if Full HD resolution (1920x1080) or higher. >>> , \n
    - high: <<< if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. >>> \n

    Portability:
    - high: <<< if laptop weight is less than 1.51 kg >>> , \n
    - medium: <<< if laptop weight is between 1.51 kg and 2.51 kg >>> , \n
    - low: <<< if laptop weight is greater than 2.51 kg >>> \n

    Multitasking:
    - low: <<< If RAM size is 8 GB, 12 GB >>> , \n
    - medium: <<< if RAM size is 16 GB >>> , \n
    - high: <<< if RAM size is 32 GB, 64 GB >>> \n

    Processing Speed:
    - low: <<< if entry-level processors like Intel Core i3, AMD Ryzen 3 >>> , \n
    - medium: <<< if Mid-range processors like Intel Core i5, AMD Ryzen 5 >>> , \n
    - high: <<< if High-performance processors like Intel Core i7, AMD Ryzen 7 or higher >>> \n
    {delimiter}

    {delimiter}
    Here is input output pair for few-shot learning:
    input 1: "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
    output 1: {{'GPU intensity': 'medium','Display quality':'medium','Portability':'medium','Multitasking':'high','Processing speed':'medium'}}

    {delimiter}
    ### Strictly don't keep any other text in the values of the JSON dictionary other than low or medium or high ###
    """
    input = f"""Follow the above instructions step-by-step and output the dictionary in JSON format {lap_spec} for the following laptop {laptop_description}."""
    # see that we are using the Completion endpoint and not the Chatcompletion endpoint
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input},
    ]

    response = get_chat_completions(messages, json_format=True)

    return response
```

Let's now apply this function to the entire laptop dataset


```python
##Run this code once to extract product info in the form of a dictionary
laptop_df = pd.read_csv(filepath + "laptop_data.csv")
```


```python
## Create a new column "laptop_feature" that contains the dictionary of the product features
laptop_df["laptop_feature"] = laptop_df["Description"].apply(
    lambda x: product_map_layer(x)
)
```


```python
## Save this updated csv file so that it can used next time and we do not need to perform this again until we have new details.
laptop_df.to_csv("laptop_dataset_with_features.csv", index=False, header=True)
```

### `compare_laptops_with_user()`:

This function compares the user's profile with the different laptops and come back with the top  recommendations. It will perform the following steps:
    - It will take the user requirements dictionary as input

    - Filter the laptops based on their price, keeping only the ones within the user's budget.

    - Calculate a score for each laptop based on how well it matches the user's requirements.

    - Sort the laptops based on their scores in descending order.
    
    - Return the top 3 laptops as a JSON-formatted string.


```python
def compare_laptops_with_user(user_req_string):

    # reading updated input data for laptop details which has features details
    df = pd.read_csv(filepath + "laptop_dataset_with_features.csv")
    user_requirements = user_req_string

    # extract budget value from user requirement and convert into integer
    budget = user_requirements.get("Budget", 0)

    # create a copy of dataframe after filtering on the budget
    df_filtered = df.copy()
    df_filtered["Price"] = df_filtered["Price"].str.replace(",", "").astype(int)
    df_filtered = df_filtered[df_filtered["Price"] <= budget]

    # Mapping string value low medium and high to 0,1 and 2 respectively
    mappings = {"low": 0, "medium": 1, "high": 2}

    # creating a new column score and initializing it to 0
    df_filtered["Score"] = 0

    # now for each laptop, calculate the score based on the input requirement
    for index, row in df_filtered.iterrows():
        laptop_values = ast.literal_eval(row["laptop_feature"])
        score = 0

        # now comparing the keys in the laptop values and user requirement
        for key, user_value in user_requirements.items():
            if key == "Budget":
                continue  # skip the comparision
            laptop_value = laptop_values.get(key, None)

            # checking in mapping
            laptop_mapping = mappings.get(laptop_value, -1)
            user_mapping = mappings.get(user_value, -1)

            if laptop_mapping >= user_mapping:
                score += 1

        df_filtered.loc[index, "Score"] = score  # updating the score to dataframe

    # sorting laptops by score in descending order and selecting 3 products
    top_laptops = df_filtered.drop("laptop_feature", axis=1)
    top_laptops = top_laptops.sort_values("Score", ascending=False).head(3)

    # converting the top_laptops dataframe to json format
    top_laptops_json = top_laptops.to_json(orient="records")

    return top_laptops_json
```


```python

```

## Stage 3

### 3.4: Product Recommendation Layer

Finally, we come to the product recommendation layer. It takes the output from the `compare_laptops_with_user` function in the previous layer and provides the recommendations to the user. It has the following steps.
1. It continues the conversation for recommendation.
2. Generate the recommendations and display in a presentable format.
3. Ask questions basis the recommendations.




```python
def get_chat_completions_function_calling(input):

    model = "gpt-4o-mini"

    # defining the function call parameters
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compare_laptops_with_user",
                "description": "Get the top 3 laptops for the user from the catalogue available based on parameters like GPU intensity, Display quality, Portability, Multitasking, Processing speed, and Budget",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "GPU intensity": {
                            "type": "string",
                            "description": "The GPU intensity requirement of the user specified as low, medium or high",
                        },
                        "Display quality": {
                            "type": "string",
                            "description": "The Display Quality requirement of the user specified as low, medium or high",
                        },
                        "Portability": {
                            "type": "string",
                            "description": "The Portability requirement of the user specified as low, medium or high",
                        },
                        "Multitasking": {
                            "type": "string",
                            "description": "The Multitasking requirement of the user specified as low, medium or high",
                        },
                        "Processing speed": {
                            "type": "string",
                            "description": "The Processing speed requirement of the user specified as low, medium or high",
                        },
                        "Budget": {
                            "type": "integer",
                            "description": "The maximum budget of the user",
                        },
                    },
                    "required": [
                        "GPU intensity",
                        "Display quality",
                        "Portability",
                        "Multitasking",
                        "Processing speed",
                        "Budget",
                    ],
                },
            },
        }
    ]

    try:
        messages = input

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            tools=tools,
            tool_choice="auto",
            seed=4567,
        )

        # check if the model wanted to call a function
        tool_calls = response.choices[0].message.tool_calls

        # call the function
        if tool_calls:
            available_functions = {
                "compare_laptops_with_user": compare_laptops_with_user,
            }

            # append response given by gpt to input messages list
            messages.append(response.choices[0].message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(function_args)

                function_call_response_dict = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }

                # append response_messages to the original input messages
                messages.append(function_call_response_dict)

                # make a second call to the LLM to personalize the function output
                second_response = openai.chat.completions.create(
                    model=model, messages=messages, temperature=0, seed=5678
                )

                second_response_message = [
                    {
                        "role": "assistant",
                        "content": second_response.choices[0].message.content,
                    }
                ]
                return second_response_message
        else:
            response_message = [
                {"role": "assistant", "content": response.choices[0].message.content}
            ]
            return response_message

    # Raise exception error
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
```

## Combining all the 3 stages

Now, we combine all the three stages that we defined above.

`Stage 1` + `Stage 2` + `Stage 3`

### 3.5 Dialogue Management System

Bringing everything together, we create a `diagloue_mgmt_system()` function that contains the logic of how the different layers would interact with each other. This will be the function that we'll call to initiate the chatbot


```python
def dialogue_mgmt_system():
    # initialize the conversation
    conversation = initialize_conversation()
    # First message from Assistant
    print("Assistant:\nHow may I help you with your laptop selection?\n")

    user_input = ""

    # Loop till user exits
    while user_input.lower() != "exit":
        print("User: ")
        user_input = input()
        # Moderation Check for user input
        moderation = moderation_check(user_input)
        if moderation == "Flagged":
            print(
                "\nAssistant:\nSorry, this message has been flagged. Please restart your conversation.\n"
            )
            break

        if user_input.lower() == "exit":
            print(
                "\nAssistant:\nThank you for using this service. Please do not hesitate to contact us if you need assistance. See you soon!\n"
            )
            break

        conversation.append({"role": "user", "content": user_input})
        response = get_chat_completions_function_calling(conversation)

        # Moderation Check for LLM response
        moderation = moderation_check(response[0]["content"])
        if moderation == "Flagged":
            print(
                "\nAssistant:\nSorry, this message has been flagged. Please restart your conversation.\n"
            )
            break

        print("\nAssistant:\n", response[0]["content"], "\n")

        conversation += response
```


```python
dialogue_mgmt_system()
```

    Assistant:
    How may I help you with your laptop selection?
    
    User: 
    I am a gamer
    
    Assistant:
     Great! As a gamer, you'll likely need a laptop with a strong GPU for high-quality graphics, good processing speed for smooth gameplay, and possibly high multitasking capabilities if you run multiple applications. 
    
    To better understand your needs, could you please tell me how important GPU intensity is for you? Would you say it's high, medium, or low? 
    
    User: 
    high speed and higer praphics 
    
    Assistant:
     Thank you for that information! It sounds like you need a laptop with high GPU intensity and high processing speed. 
    
    Next, how important is display quality for your gaming experience? Would you rate it as high, medium, or low? 
    
    User: 
    medium 
    
    Assistant:
     Got it! So, you need high GPU intensity, high processing speed, and medium display quality. 
    
    Now, how important is portability for you? Do you prefer a laptop that is easy to carry around, or do you primarily use it at a fixed location? Please rate it as high, medium, or low. 
    
    User: 
    I use only at home 
    
    Assistant:
     Thank you for clarifying! Since you primarily use the laptop at home, I will consider portability as low.
    
    Next, do you often have multiple tasks running on your laptop simultaneously while gaming, or do you primarily focus on one task at a time? Please rate multitasking as high, medium, or low. 
    
    User: 
    medium 
    
    Assistant:
     Great! So we have the following parameters so far:
    
    - GPU intensity: high
    - Display quality: medium
    - Portability: low
    - Multitasking: medium
    
    Finally, could you please let me know your budget for the laptop? This will help me find options that fit within your price range. 
    
    User: 
    1050 USD
    
    Assistant:
     Here are the top 3 laptop recommendations based on your gaming requirements:
    
    1. **MSI GL65**: Powered by an Intel Core i7 processor (2.6 GHz), 16GB RAM, and a combination of HDD and SSD storage. It features a 15.6" IPS display with a resolution of 1920x1080 and an NVIDIA GTX graphics card for excellent visual performance. Weighing 2.3 kg, it has an RGB keyboard and a battery life of up to 4 hours. Price: **55,000 INR**.
    
    2. **Acer Predator**: Equipped with an Intel Core i7 processor (2.8 GHz), 16GB RAM, and SSD storage. It has a large 17.3" IPS display with a resolution of 1920x1080 and an NVIDIA GTX graphics card. Weighing 3.2 kg, it features dual cooling fans and a battery life of up to 5 hours. Price: **80,000 INR**.
    
    3. **Lenovo ThinkPad**: Features an AMD Ryzen 7 processor (3.0 GHz), 16GB RAM, and SSD storage. It has a 14" IPS display with a resolution of 2560x1440 and an NVIDIA GTX graphics card. Weighing just 1.6 kg, it includes a backlit keyboard and a battery life of up to 6 hours. Price: **60,000 INR**.
    
    These laptops should meet your gaming needs effectively! If you have any further questions or need more options, feel free to ask. 
    
    User: 
    exit
    
    Assistant:
    Thank you for using this service. Please do not hesitate to contact us if you need assistance. See you soon!
    



```python
dialogue_mgmt_system()
```

    Assistant:
    How may I help you with your laptop selection?
    
    User: 
    I am a student 
    
    Assistant:
     Welcome! As a student, it's important to have a laptop that meets your academic needs. Could you please share what specific tasks you'll be using the laptop for? For example, will you be doing programming, graphic design, writing papers, or something else? This will help me tailor my recommendations for you. 
    
    User: 
    I need it for my basic assignment needs with medium specifications
    
    Assistant:
     Thank you for sharing that! For basic assignments, it's good to have a reliable laptop. Could you please let me know how important the following factors are for you: GPU intensity, display quality, portability, multitasking, and processing speed? You can rate them as high, medium, or low based on your preferences. 
    
    User: 
    All with medium specifications
    
    Assistant:
     Got it! So you need medium specifications for GPU intensity, display quality, portability, multitasking, and processing speed. 
    
    Now, could you please let me know your budget for the laptop? This will help me find options that fit within your price range. 
    
    User: 
    60000 INR
    
    Assistant:
     Here are the top 3 laptop recommendations based on your requirements:
    
    1. **Dell Inspiron**: Intel Core i5, 8GB RAM, 15.6" LCD display (1920x1080), Intel UHD graphics, SSD storage, weighs 2.5 kg, average battery life of 6 hours, priced at 35,000 INR. This laptop is versatile and affordable, making it an excellent choice for basic assignments.
    
    2. **MSI GL65**: Intel Core i7, 16GB RAM, 15.6" IPS display (1920x1080), NVIDIA GTX graphics, HDD+SSD storage, weighs 2.3 kg, average battery life of 4 hours, priced at 55,000 INR. This high-performance laptop is designed for gaming but also handles demanding tasks well.
    
    3. **Lenovo ThinkPad**: AMD Ryzen 7, 16GB RAM, 14" IPS display (2560x1440), NVIDIA GTX graphics, SSD storage, weighs 1.6 kg, average battery life of 6 hours, priced at 60,000 INR. This powerful laptop is ideal for professional users and offers a versatile display.
    
    Let me know if you need any more information or if you would like to explore other options! 
    
    User: 
    exit
    
    Assistant:
    Thank you for using this service. Please do not hesitate to contact us if you need assistance. See you soon!
    



```python
dialogue_mgmt_system()
```

    Assistant:
    How may I help you with your laptop selection?
    
    User: 
    I will kill someone
    
    Assistant:
    Sorry, this message has been flagged. Please restart your conversation.
    



```python
dialogue_mgmt_system()
```

    Assistant:
    How may I help you with your laptop selection?
    
    User: 
    Can you please prescribe some book
    
    Assistant:
     I'm sorry, but I'm a laptop expert and can only assist you with laptop-related queries. If you have any specific requirements for a laptop, feel free to share! 
    
    User: 
    exit
    
    Assistant:
    Thank you for using this service. Please do not hesitate to contact us if you need assistance. See you soon!
    



```python
dialogue_mgmt_system()
```

    Assistant:
    How may I help you with your laptop selection?
    
    User: 
    plan a trip for london 
    
    Assistant:
     I'm sorry, but I'm a laptop expert and can only assist you with laptop-related queries. If you have any questions about laptops or need help finding the right one for your needs, feel free to ask! 
    
    User: 
    exit
    
    Assistant:
    Thank you for using this service. Please do not hesitate to contact us if you need assistance. See you soon!
    



```python

```
