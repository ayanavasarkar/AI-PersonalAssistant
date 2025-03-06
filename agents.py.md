# Agents Class Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Class `Agents`](#2-class-agents)
    * [2.1 `__init__(self, model)`](#21-__init__self-model)
    * [2.2 `agent_extractPersonalInfo(self)`](#22-agent_extractpersonalinfoself)
    * [2.3 `personal_detail_comparison(self)`](#23-personal_detail_comparisonself)
    * [2.4 `agent_classifyPrompt(self)`](#24-agent_classifypromptself)
    * [2.5 `update_category(self)`](#25-updatecategoryself)
    * [2.6 `extract_category(self)`](#26-extractcategoryself)
    * [2.7 `delete_category(self)`](#27-deletecategoryself)


## 1. Introduction

This document details the `Agents` class, which provides methods for creating various AI agents using the `crewai` library.  Each agent is configured for a specific task related to personal information extraction, comparison, prompt classification, and data manipulation.


## 2. Class `Agents`

This class utilizes a language model (`model`) to power its agent creation methods.

### 2.1 `__init__(self, model)`

Initializes the `Agents` class.

| Parameter | Type | Description |
|---|---|---|
| `model` | Language Model | The language model to be used by the agents. |


### 2.2 `agent_extractPersonalInfo(self)`

Creates an agent for extracting personal information from unstructured data.

**Return Value:** An `Agent` object configured for personal information extraction.

**Algorithm:** This function instantiates an `Agent` object from the `crewai` library.  The agent is configured with the following parameters:

*   **`role`:** "Personal Information Extraction Agent"
*   **`goal`:**  Clearly instructs the agent to extract personal details and preferences without adding any information.
*   **`backstory`:** Sets the agent's context as an expert in extracting personal details.
*   **`llm`:** The language model provided during class initialization.
*   **`verbose`:** Set to `True` for detailed output during execution.
*   **`allow_delegation`:** Set to `False` to prevent the agent from delegating tasks.
*   **`max_iter`:** Limits the agent's iterative processing to 5 attempts.


### 2.3 `personal_detail_comparison(self)`

Creates an agent for comparing personal details between old and new data.

**Return Value:** An `Agent` object configured for comparing data and identifying differences.

**Algorithm:**  Similar to `agent_extractPersonalInfo`, this function creates an `Agent` with specific instructions for comparing two datasets ("old_data" and "new_data"). The agent's goal is explicitly defined to identify differences in personal details and preferences.  Incentives (rewards and penalties) are included in the goal to encourage accuracy.


### 2.4 `agent_classifyPrompt(self)`

Creates an agent for classifying user prompts into predefined categories.

**Return Value:** An `Agent` object configured for prompt classification.

**Algorithm:** This function creates an agent that categorizes user prompts into predefined categories (saving, deducing, updating, deleting memory, or off-topic). The categories are clearly defined within the agent's `goal` parameter.


### 2.5 `update_category(self)`

Creates an agent for extracting update details from user prompts and updating the data.

**Return Value:** An `Agent` object configured to identify and update data based on user input.

**Algorithm:** This agent acts as a personal assistant, interpreting user prompts to identify specific details or categories for updating within a database.  It's instructed to extract the update information and make the changes; if the prompt is unclear, it should indicate its inability to provide an answer.


### 2.6 `extract_category(self)`

Creates an agent for extracting details or categories to update or delete based on user prompts.

**Return Value:** An `Agent` object configured to extract details for updating or deleting.

**Algorithm:** This agent focuses on extracting the specific details or categories that a user wants to update or delete based on their prompt. It is explicitly instructed not to fabricate answers if it doesn't understand the input.


### 2.7 `delete_category(self)`

Creates an agent for identifying and deleting a specific category or detail from the data.

**Return Value:** An `Agent` object configured to delete specific data based on user input.

**Algorithm:** This agent functions as a personal assistant to understand user requests for data deletion. It analyzes prompts to determine what should be deleted and removes the specified portion from the provided data.  Similar to other agents, it's instructed to avoid fabricating responses if it's unable to understand the prompt.
