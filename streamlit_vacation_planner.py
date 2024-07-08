import streamlit as st
import os
import re
from secret_key import openapi_key
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

# Setting the OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = openapi_key

# Initialize the language model with a specified temperature
llm = OpenAI(temperature=0.7)

# Create prompt templates for each step
times_prompt_template = PromptTemplate(
    input_variables=['city'],
    template="What are the best times of year to visit {city}?"
)

attractions_prompt_template = PromptTemplate(
    input_variables=['city', 'best_time'],
    template="Given that the best time to visit {city} is {best_time}, can you suggest the top attractions to see?"
)

hotels_prompt_template = PromptTemplate(
    input_variables=['city', 'best_time'],
    template="Can you recommend some good places to stay in {city} during the best time to visit, which is {best_time}?"
)

food_prompt_template = PromptTemplate(
    input_variables=['city', 'best_time'],
    template="What are some popular local foods to try in {city} during the best time to visit, which is {best_time}?"
)

events_prompt_template = PromptTemplate(
    input_variables=['city', 'best_time'],
    template="Are there any events or festivals happening in {city} during the best time to visit, which is {best_time}?"
)

shopping_prompt_template = PromptTemplate(
    input_variables=['city', 'best_time'],
    template="Where can I find the best shopping areas in {city} during the best time to visit, which is {best_time}?"
)

# Create prompt templates for cost estimation
flight_cost_prompt_template = PromptTemplate(
    input_variables=['city'],
    template="What is the average cost of a flight to {city} from New York?"
)

accommodation_cost_prompt_template = PromptTemplate(
    input_variables=['city'],
    template="What is the average daily cost of accommodation in {city}?"
)

food_cost_prompt_template = PromptTemplate(
    input_variables=['city'],
    template="What is the average daily cost of food in {city}?"
)

activities_cost_prompt_template = PromptTemplate(
    input_variables=['city'],
    template="What is the average daily cost of activities in {city}?"
)

# Create LLMChains for each template
times_chain = LLMChain(llm=llm, prompt=times_prompt_template, output_key="best_time")
attractions_chain = LLMChain(llm=llm, prompt=attractions_prompt_template, output_key="top_attractions")
hotels_chain = LLMChain(llm=llm, prompt=hotels_prompt_template, output_key="hotels")
food_chain = LLMChain(llm=llm, prompt=food_prompt_template, output_key="food")
events_chain = LLMChain(llm=llm, prompt=events_prompt_template, output_key="events")
shopping_chain = LLMChain(llm=llm, prompt=shopping_prompt_template, output_key="shopping")

flight_cost_chain = LLMChain(llm=llm, prompt=flight_cost_prompt_template, output_key="flight_cost")
accommodation_cost_chain = LLMChain(llm=llm, prompt=accommodation_cost_prompt_template, output_key="accommodation_cost")
food_cost_chain = LLMChain(llm=llm, prompt=food_cost_prompt_template, output_key="food_cost")
activities_cost_chain = LLMChain(llm=llm, prompt=activities_cost_prompt_template, output_key="activities_cost")

# Load tools relevant for vacation planning
tools = load_tools(["wikipedia"], llm=llm)

# Initialize the agent with the necessary tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Function to extract cost from response using regex
def extract_cost(response_text):
    costs = re.findall(r'\$\d+', response_text)
    costs = [int(cost.strip('$')) for cost in costs]
    if costs:
        return sum(costs) / len(costs)
    else:
        return 0  # Default to 0 if no costs are found

# Streamlit app
st.title("Vacation Planner")

# Sidebar for filtering options
st.sidebar.title("Filter Recommendations")
show_attractions = st.sidebar.checkbox("Top Attractions", True)
show_hotels = st.sidebar.checkbox("Places to Stay", True)
show_food = st.sidebar.checkbox("Food to Try", True)
show_events = st.sidebar.checkbox("Events and Festivals", True)
show_shopping = st.sidebar.checkbox("Shopping Areas", True)

# Prompt the user for input
user_input = st.text_input("Enter a city:")

if st.button("Plan my vacation"):
    # Run the chains with the user's input for the 'city' variable
    if user_input:
        # Run the times chain first to get the best time to visit
        best_time_response = times_chain({"city": user_input})
        best_time = best_time_response["best_time"]

        # Initialize the response dictionary
        response = {"best_time": best_time}

        # Conditionally run other chains based on user selections
        if show_attractions:
            attractions_response = attractions_chain({"city": user_input, "best_time": best_time})
            response["top_attractions"] = attractions_response["top_attractions"]

        if show_hotels:
            hotels_response = hotels_chain({"city": user_input, "best_time": best_time})
            response["hotels"] = hotels_response["hotels"]

        if show_food:
            food_response = food_chain({"city": user_input, "best_time": best_time})
            response["food"] = food_response["food"]

        if show_events:
            events_response = events_chain({"city": user_input, "best_time": best_time})
            response["events"] = events_response["events"]

        if show_shopping:
            shopping_response = shopping_chain({"city": user_input, "best_time": best_time})
            response["shopping"] = shopping_response["shopping"]

        # Estimate costs using the Langchain agent
        flight_cost_response = flight_cost_chain({"city": user_input})
        accommodation_cost_response = accommodation_cost_chain({"city": user_input})
        food_cost_response = food_cost_chain({"city": user_input})
        activities_cost_response = activities_cost_chain({"city": user_input})

        flight_cost = extract_cost(flight_cost_response["flight_cost"])
        accommodation_cost = extract_cost(accommodation_cost_response["accommodation_cost"])
        food_cost = extract_cost(food_cost_response["food_cost"])
        activities_cost = extract_cost(activities_cost_response["activities_cost"])

        # Calculate the total estimated cost for a week-long stay and a 3-day weekend
        total_week_cost = (
            flight_cost +
            (accommodation_cost * 7) +
            (food_cost * 7) +
            (activities_cost * 7)
        )

        total_weekend_cost = (
            flight_cost +
            (accommodation_cost * 3) +
            (food_cost * 3) +
            (activities_cost * 3)
        )

        # Display the results
        st.write("Estimated total cost for a week-long stay: $", total_week_cost)
        st.write("Estimated total cost for a 3-day weekend stay: $", total_weekend_cost)

        st.write("Best time to visit:", response["best_time"])

        if show_attractions:
            st.write("Top attractions:", response.get("top_attractions", "Not available"))

        if show_hotels:
            st.write("Places to stay:", response.get("hotels", "Not available"))

        if show_food:
            st.write("Food to try:", response.get("food", "Not available"))

        if show_events:
            st.write("Events and festivals:", response.get("events", "Not available"))

        if show_shopping:
            st.write("Shopping areas:", response.get("shopping", "Not available"))

    else:
        st.write("Please enter a city.")
