from crewai import Crew, Process

from agents import Agents
from tasks import Tasks
from langchain_core.documents import Document

class AI_Agents():
    """
    A class to manage AI agent tasks and workflows using CrewAI framework.
    """

    def __init__(self, model, output_path: str) -> None:
        """
        Initializes the AI_Agents class with the given model and output path.

        Args:
            model (ChatGroq): The model to be used by the agents.
            output_path (str): The path where output files will be saved.
        """
        self.model = model
        self.agent = Agents(model=model)
        self.tasks = Tasks(path=output_path)

    def extract_from_uploaded_file(self, file: str) -> str:
        """
        Extracts data from the uploaded file and processes it to extract personal information.

        Args:
            file (str): The uploaded file to be processed.

        Returns:
            The extracted personal information from the file.
        """
        file_ext = file.name.split(".")[-1]
        
        if file_ext == "txt":
            data = file.read().decode()
        else:
            RuntimeWarning

        data_agent = self.agent.agent_extractPersonalInfo()
        # Kick off the crew's work
        data_task = self.tasks.task_personalInfo(data, data_agent)

        # Instantiate your crew with a sequential process
        data_crew = Crew(
            agents=[data_agent],
            tasks=[data_task],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
        results = data_crew.kickoff()
        return self.output(results)

    def prompt_classifier(self, prompt: str) -> str:
        """
        Classifies the given prompt into predefined categories.

        Args:
            prompt (str): The user prompt to be classified.

        Returns:
            The classification result of the prompt.
        """
        agent_prompt_classifier = self.agent.agent_classifyPrompt()
        task_prompt_classifier = self.tasks.task_classifyPrompt(prompt, agent_prompt_classifier)

        crew_prompt = Crew(
            agents=[agent_prompt_classifier],
            tasks=[task_prompt_classifier],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
        prompt_result = crew_prompt.kickoff()
        return self.output(prompt_result)
    
    def memory_management(self, prompt: str, db_data: Document) -> str:
        """
        Manages memory by updating it based on the given prompt and database data.

        Args:
            prompt (str): The user prompt for updating memory.
            db_data (Document): The existing data from the database to be updated.

        Returns:
            The updated memory data.
        """
        agent_memory = self.agent.update_category()
        task_memory = self.tasks.task_update_category(prompt=prompt, data=db_data.page_content, agent=agent_memory) 
        crew_memory = Crew(
            agents=[agent_memory],
            tasks=[task_memory],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
        r = crew_memory.kickoff()
        return self.output(r)
    
    def category_extraction(self, prompt: str) -> str:
        """
        Extracts the category or detail the user wants to update or delete from the prompt.

        Args:
            prompt (str): The user prompt indicating the category or detail.

        Returns:
            The extracted category or detail from the prompt.
        """
        extraction_agent = self.agent.extract_category()
        extraction_task = self.tasks.task_extract_category(prompt, extraction_agent)

        extraction_crew = Crew(
            agents=[extraction_agent],
            tasks=[extraction_task],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )

        extracted_class = extraction_crew.kickoff()
        return self.output(extracted_class)
    
    def delete_memory(self, prompt: str, db_data: Document) -> str:
        """
        Deletes a specific category or detail from the database based on the user prompt.

        Args:
            prompt (str): The user prompt indicating what to delete.
            db_data (Document): The existing data from the database to be processed for deletion.

        Returns:
            The updated data after deletion.
        """
        deletion_agent = self.agent.delete_category()
        deletion_task = self.tasks.task_delete_category(prompt, db_data.page_content, deletion_agent)

        deletion_crew = Crew(
            agents=[deletion_agent],
            tasks=[deletion_task],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
        res = deletion_crew.kickoff()
        return self.output(res)

    def output(self, result: dict) -> str:
        """
        Extracts and returns the final output from the task result.

        Args:
            result: The result obtained from the task execution.

        Returns:
            The final output of the task.
        """
        return result['final_output']