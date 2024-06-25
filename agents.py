from crewai import Agent

class Agents:
    def __init__(self, model) -> None:
        """
        Initialize the Agents class with a model.

        Args:
            model: The language model to be used by the agents.
        """
        self.model = model 

    def agent_extractPersonalInfo(self):
        """
        Create an agent for extracting personal information from unstructured data.

        Returns:
            Agent: An agent configured to extract personal details and preferences from unstructured information.

        Description:
            This method returns an Agent designed to take in unstructured information about a person and extract all 
            personal details and preferences. The agent is set up with a specific role, goal, backstory, and additional 
            settings such as verbosity and iteration limits.
        """
        return Agent(
            role='Personal Information Extraction Agent',
            goal='take in unstructured information about a person and extract all personal details\
                and preferences from the user from that unstructured data. Do not add any extra string. Just extract what is already there.',
            backstory='You are a master at extracting the necessary personal details about a person from the information entered',
            llm = self.model,
            verbose = True,
            allow_delegation=False,
            max_iter=5,
            # memory=True,
        )

    def personal_detail_comparison(self):
        """
        Create an agent for comparing personal details between old and new data.

        Returns:
            Agent: An agent configured to compare old and new data to identify differences.

        Description:
            This method returns an Agent designed to compare old data with new data and identify differences. The agent 
            is tasked with finding discrepancies between the two datasets, focusing on personal details and preferences. 
            The agent operates under a specific role, goal, and backstory, with verbosity and iteration limits.
        """
        return Agent(
            role='Difference Identifier',
            goal="""take in two sources of data - old_data, which is the previous data and new_data, which is the new data entered. \
                both old_data and new_data contain personal details and preferences about the same person. \
                your goal is to compare the new_data with old_data and find the following things: \
                1. differences between the new data and the old data. \
                \
                I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or 
                add any incorrect information.
                Take a deep breath, think step by step, and then do the comparison.""",
            backstory = """You are a master at comparing old data with new data and identifying what must be updated based on the new information.""",
            llm = self.model,
            verbose = True,
            allow_delegation=False,
            max_iter=5,
        )

    def agent_classifyPrompt(self):
        """
        Create an agent for classifying user prompts into predefined categories.

        Returns:
            Agent: An agent configured to categorize user prompts based on their intent.

        Description:
            This method returns an Agent designed to take user prompts and categorize them into one of several predefined 
            categories, such as saving, deducing, updating, or deleting memory, or identifying off-topic prompts. The agent 
            is set up with a specific role, goal, backstory, and additional settings such as verbosity and iteration limits.
        """
        return Agent(
            role='User Prompt Classification Agent',
            goal='take in user prompt from a human and categorize it into one of the following categories: \
                save something in memory - when someone wants to save the previous data \
                deduce memory from unstructured text - when someone wants to extract information from stored data in memory \
                update memory - when someone wants to update the already stored information in the memory \
                delete memory - when someone wants to delete the entire information stored in memory \
                off_topic when it doesnt relate to any other category',
            
            backstory='You are a master at understanding what the user wants when they \
                they communicate with you and are able to categorize it in a useful way',
            llm = self.model,
            verbose = True,
            allow_delegation=False,
            max_iter=5,
            # memory=True,
        )
    
    def update_category(self):
        """
        Create an agent for extracting update details from user prompts and update that part in the data.

        Returns:
            Agent: An agent configured to identify and update specific details or categories to be updated based on user input.

        Description:
            This method returns an Agent designed to understand what the user wants to update in a database based on the 
            entered prompt. The agent should accurately extract the detail or category the user intends to update, and if 
            the prompt is unclear, it should indicate that it cannot provide an answer. Then it updates that part of the data.
        """
        return Agent(
            role='Personal Assistant',
            goal="""you are a personal assistant who understands what the user wants to update in a database based on the prompt entered. \
                    you would be given the database data as string and the user query prompt.
                    Your goal is to extract the specific detail or category that the user wants to update and make the change in the data given. \
                    If you do not understand the prompt; do not make up answers. \
                    Simply say that 'You do not have information to answer the question """,
            verbose=True,
            # memory=True,
            llm = self.model,
            max_iter=5,
            backstory=(
                "You are the best personal assistant who figure out what detail the user wants to update from the query entered by the \
                    user and update the necessary part of the data. You do not make up answers if you do not know them."
            ),
            # tools=[],
            allow_delegation=False,
        )

    def extract_category(self):
        """
        Create an agent for extracting details or categories to update or delete based on user prompts.

        Returns:
            Agent: An agent configured to extract specific details or categories to be updated or deleted from user input.

        Description:
            This method returns an Agent designed to understand and extract the specific detail or category the user wants 
            to update or delete based on the prompt. The agent should accurately identify the detail or category and its 
            new value if present. If the agent cannot understand the prompt, it should indicate that it cannot provide an answer.
        """
        return Agent(
            role='Detail Extractor',
            goal="""you are a personal assistant who understands what the user wants to delete in a database based on the prompt entered. \
                    you would be given the user query prompt and your goal is to extract the specific detail or category \
                    that the user wants to delete with the value if it is present. \
                    If you do not understand the prompt; do not make up answers. \
                    Simply say that 'You do not have information to answer the question """,
            verbose=True,
            # memory=True,
            llm = self.model,
            max_iter=5,
            backstory=(
                "You are the best personal assistant who can figure out what detail the user wants to delete from the query entered by the \
                    user. You do not need to do anything except extracting the detail or category. You do not make up answers if you do not know them."
            ),
            # tools=[],
            allow_delegation=False,
        )
    
    def delete_category(self):
        """
        Create an agent for identifying and deleting a specific category or detail from the data based on user input.

        Returns:
            Agent: An Agent object configured to delete a specific category or detail from the data.

        Description:
            This method creates an Agent to understand the user's prompt and identify the specific category or detail 
            to be deleted from the provided data. The agent's goal is to analyze the user's query to determine what to 
            delete, remove the relevant part of the data, and then return the data without the specified part. If the 
            agent cannot understand the user's intent, it should indicate that it cannot provide an answer.
        """
        return Agent(
            role='Personal Assistant',
            goal="""you are a personal assistant who understands what the user wants to delete in a database based on the prompt entered. \
                    you would be given the database data as string and the user query prompt.
                    Your goal is to extract the specific detail or category that the user wants to delte and remove that portion in the data given. \
                    If you do not understand the prompt; do not make up answers. \
                    Simply say that 'You do not have information to answer the question """,
            verbose=True,
            # memory=True,
            llm = self.model,
            max_iter=5,
            backstory=(
                "You are the best personal assistant who figure out what detail or information the user wants to delete from the query entered by the \
                    user and remove the necessary part of the data. You do not make up answers if you do not know them."
            ),
            # tools=[],
            allow_delegation=False,
        )