from crewai import Task
from agents import Agents

class Tasks():
    def __init__(self, path: str) -> None:
        """Set the output path"""
        self.path = path

    def task_personalInfo(self, data: str, agent: Agents):
        """
        Analyze the provided data to extract personal information.

        Args:
            data (str): The data to be analyzed for extracting personal information.
            agent (Agents): The agent responsible for performing the task.

        Returns:
            Task: A Task object configured to analyze the data and extract personal information.

        Description:
            This method creates a Task to conduct a comprehensive analysis of the provided data to extract personal 
            information about the user, including personal choices, preferences, and other relevant details. The task 
            expects an output in the form of bullet points for each category of personal information found. If no 
            useful information is identified, the task should return clear instructions indicating this.
        """
        return Task(
            description=f"""Conduct a comprehensive analysis of the data provided and extract \
                personal information about the user from the data. \
                    Extract any personal choices or preferences and other things. \
                        
                    DATA:\n\n {data} \n\n""",

            expected_output="""A set of bullet points for each category of the personal information. \
                If no information is found then return clear instructions that no useful material was found.""",
            # expected_output="""A nested json where the key is the category of the personal information \
            # and the value is a necessary information about the person.
            # If no information is found then return clear instructions that no useful material was found.""",
            
            # Example:
            # {'Contact Information': {'Email': 'test@gmail.com'}, {'Phone Number': '123456789'}}
            # """,
            output_file=self.path+"/extracted_info.txt",
            agent=agent,
            )

    def task_personal_detail_comparison(self, old_data: str, new_data: str, agent: Agents):
        """
        Compare old and new data to identify differences.

        Args:
            old_data (str): The old data to be compared.
            new_data (str): The new data to be compared.
            agent (Agents): The agent responsible for performing the task.

        Returns:
            Task: A Task object configured to compare the old and new data and identify differences.

        Description:
            This method creates a Task to conduct a comprehensive comparison between the old and new data. The task 
            aims to identify and list the differences between the two datasets, with a focus on highlighting changes 
            in various categories. The output should be formatted as bullet points, indicating the category, new data 
            point, and old data point where differences are found.
        """

        return Task(
            description=f""" Conduct a comprehensive comparison between old data {old_data} and new data {new_data}. 
            Based on the comparison, identify the following things: \
            1. differences in the old and the new data. 
            
            OLD DATA: \n\n {old_data} \n\n
            NEW DATA: \n\n {new_data} \n\n

            Use the following format during output:
            - <category> <new_data_point> <old_data_point>

            I will tip you $20 if you are perfect, and I will fine you $40 if you miss any important information or 
                add any incorrect information.
                Take a deep breath, think step by step, and then do the comparison.
            """,

            expected_output= """ A set of bullet points of categories where there is difference between old data and new data""",
            output_file=f"{self.path}/difference_info.txt",
            agent=agent,
        )

        
    def task_classifyPrompt(self, data: str, agent: Agents):
        """
        Classify the user prompt into a predefined category.

        Args:
            data (str): The prompt entered by the user.
            agent (Agents): The agent responsible for performing the task.

        Returns:
            Task: A Task object configured to analyze and classify the user prompt.

        Description:
            This method creates a Task to analyze the user prompt and categorize it into one of the following 
            categories: 'save something in memory', 'deduce memory from unstructured text', 'update memory', 
            'delete memory', or 'off_topic'. The task outputs a single category that best describes the user's intent 
            based on the prompt.
        """
        return Task(
            description=f"""Conduct a comprehensive analysis of the prompt entered by human and categorize it into one of the following categories: \
                save something in memory - when someone wants to save the previous data \
                deduce memory from unstructured text - when someone wants to extract information from stored data in memory \
                update memory - when someone wants to update the already stored information in the memory \
                delete memory - when someone wants to delete the entire information stored in memory \
                off_topic when it doesnt relate to any other category \
                
                USER INPUT:\n\n {data} \n\n
                Output a single cetgory only""",
            expected_output="""A single categtory for the type of prompt from the types \
                [save something in memory, deduce memory from unstructured text, update memory, delete memory, off_topic] \
                eg:
                update memory """,
            output_file=f"{self.path}/categorized_user_input.txt",
            agent=agent,
            )
    
    def task_update_category(self, prompt: str, data: str, agent: Agents):
        """
        Determine the category or detail to update based on user input.

        Args:
            prompt (str) : The user input prompt indicating what to update
            data (str): The most similar data to the prompt that must be updated.
            agent (Agents): The agent responsible for performing the task.

        Returns:
            Task: A Task object configured to determine the category or detail to be updated.

        Description:
            This method creates a Task to interpret the user's input and identify the specific category or detail they 
            wish to update based on the data in the database. Then it updates the specified part and stores the new output.
            If the task cannot understand the user's intent, it should clearly state that. The task 
            aims to produce an output that is comprehensible and actionable for updating the database.
        """
        return Task(
            description = f"""Your task is to understand what category or specific detail the user wants to update in the {data} from the {prompt} entered 
                by the user. \
                If you do not understand please say that you do not understand what the user wants.
                In short perform the following steps:
                1. Take the user entered query and figure out what category or detail the user wants to update in the data.
                2. From the entered query extract the new information to update the data with.
                3. Comprehensively analyze the data given to you and extract the exact part of the data to which you need to update.
                4. Update the part of the data.
                5. Return the entire data with the updated part.
                """,
            expected_output='Update the category or detail that the user wants to update in the data.',
            # context= [self.task_personalInfo],
            # human_input = True,
            output_file="{self.path}/update.txt",
            agent=agent
        )
    
    def task_extract_category(self, prompt: str, agent: Agents):
        """
        Extract the category or detail the user wants to update or delete or interact with.

        Args:
            prompt (str): The prompt entered by the user.
            agent (Agents): The agent responsible for performing the task.

        Returns:
            Task: A Task object configured to extract the category or detail from the user's prompt.

        Description:
            This method creates a Task to analyze the user's prompt and extract the specific category or detail the 
            user wishes to update or delete. The task should output the category or detail as a maximum of 5 words. 
            If the user's intent is unclear, the task should indicate that it does not understand.
        """
        return Task(
            description=f"""Your task is to understand what category or specific detail the user wants to delete from the {prompt} entered 
                by the user. If you do not understand please say that you do not understand what the user wants.
                In short perform the following steps:
                1. Thoroughly analye and understand the {prompt} entered by the user.
                2. Extract the category or detail that the user wants to delete.
                If the user wants to delete a category, then output just the Category as <CATEGORY>
                Just output the exact category or detail as a maximum of 5 words. Do not output a sentence.
            """,
            expected_output='Extract the exact detail or category that the user wants to delete in the data.',
            # context= [self.task_personalInfo],
            # human_input = True,
            output_file="{self.path}/extract_cat.txt",
            agent=agent
        )

    def task_delete_category(self, prompt: str, data: str, agent: Agents):
        """
        Analyze the provided data to extract personal information to be deleted.

        Args:
            prompt (str): The user prompt entered.
            data (str): The data to be analyzed for extracting personal information.
            agent (Agents): The agent responsible for performing the task.

        Returns:
            Task: A Task object configured to analyze the data and extract personal information.

        Description:
            This method creates a Task to conduct a comprehensive analysis of the provided data to extract personal 
            information about the user, including personal choices, preferences, and other relevant details. The task 
            expects an output in the form of bullet points for each category of personal information found. If no 
            useful information is identified, the task should return clear instructions indicating this.
        """
        return Task(
            description=
                f"""Your task is to understand what category or specific detail the user wants to delete in the {data} from the {prompt} entered 
                by the user. \
                If you do not understand please say that you do not understand what the user wants.
                In short perform the following steps:
                1. Take the user entered query and figure out what category or detail the user wants to delete in the data.
                2. From the entered query extract the new information to update the data with.
                3. Comprehensively analyze the data given to you and extract the exact part of the data to which you need to delete.
                4. Delete that specific part of the data.
                5. Return the entire data without the deleted part.
                """,
            expected_output='Remove the category or detail that the user wants to delete in the data.',
            # context= [self.task_personalInfo],
            # human_input = True,
            output_file="{self.path}/rmv.txt",
            agent=agent
        )
