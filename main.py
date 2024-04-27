import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from decouple import config
from textwrap import dedent
from agents import get_active_agent
from tasks import CustomTasks

# Ensure environment variables are set up
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = config("OPENAI_ORGANIZATION_ID")

class CustomCrew:
    def __init__(self):
        self.agent = get_active_agent()
        self.tasks = CustomTasks()

    def run(self):
        # Define specific tasks based on the agent's profile (family or adult)
        if self.agent.role == "Family Conversationalist":
            task = self.tasks.fetch_user_query(self.agent, "Tell me about dinosaurs.")
        else:
            task = self.tasks.provide_detailed_explanation(self.agent, "Discuss the theory of relativity.")

        # Setup the crew with the active agent and their respective task
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
        )

        # Execute the tasks and collect the results
        result = crew.kickoff()
        return result

# Main function to run the custom crew
if __name__ == "__main__":
    print("## Welcome to the Conversational Crew AI")
    print("-------------------------------")
    
    custom_crew = CustomCrew()
    result = custom_crew.run()
    print("\n\n########################")
    print("## Here is your custom crew run result:")
    print("########################\n")
    print(result)

