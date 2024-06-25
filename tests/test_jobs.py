import pytest, os
from unittest.mock import MagicMock
from crewai import Task, Agent
from agents import Agents
from tasks import Tasks
from langchain_groq import ChatGroq

from crewai import Crew, Process

class TestTasks:
    @pytest.fixture
    def agents(self):
        model = ChatGroq(
                api_key="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",
                model="llama3-70b-8192"
            )
        return Agents(model)

    @pytest.fixture
    def tasks(self):
        return Tasks(path="/tmp/")  # Adjust the path as needed for your testing environment

    def create_crew(self, agent, task):
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
        results = crew.kickoff()
        return results['final_output']

    def test_task_personalInfo(self, tasks, agents):
        """ Test the Personal Information extraction Agent.
        """
        data = "My name is John Doe."
        agent = agents.agent_extractPersonalInfo()
        task = tasks.task_personalInfo(data, agent)
        assert isinstance(task, Task)
        assert "John Doe" in task.description
        assert "extracted_info.txt" in task.output_file
        crew_res = self.create_crew(agent, task)
        assert "John Doe" in crew_res

    def test_task_classifyPrompt(self, tasks, agents):
        """ Test the task classification agent.
        """
        prompts = ["Hello", "Save in memory", "Can you update email id with test@g.com?", "Can you delete email id?"]
        outputs = ["off_topic", 'save something in memory', 'update memory', 'delete memory']
        agent = agents.agent_classifyPrompt()
        task = []
        for prompt in prompts:
            task.append(tasks.task_classifyPrompt(prompt, agent))
        
        crews = Crew(
            agents=[agent],
            tasks=task[:],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
        _ = crews.kickoff()
        assert str(task[0].output.raw_output) == outputs[0]
        assert str(task[1].output.raw_output) == outputs[1]
        assert str(task[2].output.raw_output) == outputs[2]
        assert str(task[3].output.raw_output) == outputs[3]

    def test_task_update_category(self, tasks, agents):
        prompt = "Update name with Jane Doe"
        data = "Name is John Doe."
        agent = agents.update_category()
        task = tasks.task_update_category(prompt, data, agent)
        assert isinstance(task, Task)
        crew_res = self.create_crew(agent, task)
        assert "Jane Doe" in task.description
        assert "Jane Doe" in crew_res
        assert "John Doe" not in crew_res

    def test_task_extract_category(self, tasks, agents):
        prompt = "The email id of the person is test@gmail.com"
        agent = agents.extract_category()
        task = tasks.task_extract_category(prompt, agent)
        r = self.create_crew(agent, task)
        assert isinstance(task, Task)
        assert "email id" in r.lower()
        assert prompt in task.description

    def test_task_delete_category(self, tasks, agents):
        prompt = "Delete email id"
        data = "Name - Tester Smith \n Email ID - test@g.com"
        agent = agents.delete_category()
        task = tasks.task_delete_category(prompt, data, agent)
        r = self.create_crew(agent, task)

        assert isinstance(task, Task)
        assert data in task.description
        assert prompt in task.description
        assert "test@gm.com" not in task.output.raw_output