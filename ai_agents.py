from crewai import Crew, Process
import os

from agents import Agents
from tasks import Tasks


class AI_Agents():

    def __init__(self, model, output_path) -> None:
        self.model = model
        self.agent = Agents(model=model)
        self.tasks = Tasks(path=output_path)

    def extract_from_uploaded_file(self, file):
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

    def prompt_classifier(self, prompt):
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
    
    def memory_management(self, prompt, db_data):
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
    
    def category_extraction(self, prompt):
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
    
    def delete_memory(self, prompt, db_data):
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

    def output(self, result):
        return result['final_output']