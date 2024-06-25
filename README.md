# Personal Assistant
A personal AI assistant who analyzes personal data and provides answers to questions based on the data. Following are the capabilities of the Assistant:

* Analyze any text file uploaded by the User and extract key personal details. Update the DB with the key details.
* Update any key categories or personal details in the DB as per user prompt.
* Delete any particular user information from the DB as per the user prompt.
* Answer any questions about the personal details contained in the DB.
* General chat with Groq Llama LLM.

## Brief Overview of Used Frameworks
* I have used here the open-sourced Llama model (8-billion parameter) as the base LLM for this implementation. Currently, [Groq Cloud](https://console.groq.com/playground) provides a free API for its usage on their cloud.
* I have used [CrewAI](https://www.crewai.com/) open sourced library for creating all AI-agents using their crew-based AI agent orchestration flow. 
*  CrewAI is a cutting-edge framework for orchestrating role-playing, autonomous AI agents. It is built on top on [LangChain](https://www.langchain.com/) framework.
* I have used [Chroma DB](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/) for building our Vector Database.

## Architectural Diagram and Workflow
![Alt text](https://github.com/ayanavasarkar/personal_assistant/blob/main/ui_imgs/monochrome_diagram.jpeg)

![Alt text](https://github.com/ayanavasarkar/personal_assistant/blob/main/ui_imgs/colorful_diagram.jpeg)

### Explanation of the Workflow:


### Note: Limitations
1. The system currently only accepts ```.txt``` files. Any other file will throw an Exception and crash the GUI.
2. While inputing ```.txt``` files, the system does not check if the content is already present in the DB. Hence, if duplicat entry is given, there willbe duplicates in the DB.
3. The agents have been specifically prompted to not answer anything beyond the data present in the DB. Hence, for questions requiring some amount of analysis, the system does not answer them.

## How to run the StreamLit UI

1. Load all the necessary libraries. They have been mentioned in the ```requirements.txt```.
Make sure that the versions of ```LangChain``` and ```PyDantic``` match.

2. When everything is installed, set up an account on [Groq Cloud](https://console.groq.com/playground). 

3. Create an API key for the Llama 8B parameter version from the Groq cloud account. Set up the API key on your local machine using ```export GROQ_API_KEY=<your-api-key-here>```.

4. Until the time of writing setting up and using Groq for Llama is free. This may change over time. In that case, use some other keys and make changes likewise to the model inside the file ```utils.py```.

