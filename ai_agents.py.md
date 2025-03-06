# AI_Agents Class Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Class `AI_Agents`](#2-class-ai_agents)
    * [2.1 Constructor (`__init__`) ](#21-constructor-__init__)
    * [2.2 Method `extract_from_uploaded_file`](#22-method-extract_from_uploaded_file)
    * [2.3 Method `prompt_classifier`](#23-method-prompt_classifier)
    * [2.4 Method `memory_management`](#24-method-memory_management)
    * [2.5 Method `category_extraction`](#25-method-category_extraction)
    * [2.6 Method `delete_memory`](#26-method-delete_memory)
    * [2.7 Method `output`](#27-method-output)


## 1. Introduction

This document details the `AI_Agents` class, designed to manage AI agent tasks and workflows using the CrewAI framework.  The class leverages agents and tasks defined in the `agents` and `tasks` modules, respectively, to perform various operations including data extraction, prompt classification, memory management, and data deletion.  The CrewAI framework enables parallel or sequential processing of these tasks.


## 2. Class `AI_Agents`

This class orchestrates AI agent operations using the CrewAI library.

### 2.1 Constructor (`__init__`)

```python
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
```

The constructor initializes the `AI_Agents` object. It takes a `model` (presumably a language model) and an `output_path` as input. It instantiates `Agents` and `Tasks` objects, passing the model and output path respectively, to handle agent and task management.


### 2.2 Method `extract_from_uploaded_file`

```python
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
    data_task = self.tasks.task_personalInfo(data, data_agent)

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
```

This method extracts data from a file (currently only supporting `.txt` files). It uses a CrewAI `Crew` object to execute the extraction task sequentially. The `agent_extractPersonalInfo()` method from the `Agents` class defines the agent responsible for this extraction, and `task_personalInfo()` from the `Tasks` class defines the specific task. The results are then processed using the `output` method.


### 2.3 Method `prompt_classifier`

```python
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
```

This method classifies a given prompt using a CrewAI `Crew`.  Similar to `extract_from_uploaded_file`, it uses dedicated agents and tasks (`agent_classifyPrompt`, `task_classifyPrompt`) for prompt classification.  The classification is performed sequentially.

### 2.4 Method `memory_management`

```python
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
```

This method updates memory based on a user prompt and database data.  It uses the `update_category` agent and task from `Agents` and `Tasks` respectively, executing them sequentially within a CrewAI `Crew`.


### 2.5 Method `category_extraction`

```python
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
```

This method extracts a category or detail from a user prompt. It uses a dedicated agent (`extract_category`) and task (`task_extract_category`) executed sequentially within a CrewAI `Crew`.


### 2.6 Method `delete_memory`

```python
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
```

This method deletes a category or detail from the database based on a user prompt.  It leverages the `delete_category` agent and task for this operation, running them sequentially in a CrewAI `Crew`.


### 2.7 Method `output`

```python
def output(self, result: dict) -> str:
    """
    Extracts and returns the final output from the task result.

    Args:
        result: The result obtained from the task execution.

    Returns:
        The final output of the task.
    """
    return result['final_output']
```

This helper method extracts the `final_output` from the dictionary returned by the CrewAI `Crew.kickoff()` method.  This standardizes output retrieval from different CrewAI tasks.
